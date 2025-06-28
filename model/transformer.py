import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import time
import subprocess
import pickle
from model.decoder import *
from model.optimizer import NoamOpt
from model.util import *
from model.generator import Generator, greedy, beam_search


class GATEncoder(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.1, layer=2):
        super().__init__()
        self.layer = layer
        objcnndim = 2048
        self.trans_obj = nn.Sequential(
            Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
            Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
        )
        # text
        self.mhatt_x      = clone(MultiHeadedAttention(n_heads, d_model, dropout), layer)
        self.res4mes_x    = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4ffn_x    = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ffn_x        = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        # image
        self.mhatt_o      = clone(MultiHeadedAttention(n_heads, d_model, dropout, v=0, output=0), layer)
        self.res4mes_o    = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4ffn_o    = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ffn_o        = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        # cross
        self.mhatt_x2o    = clone(Linear(d_model * 2, d_model), layer)
        self.mhatt_o2x    = clone(Linear(d_model * 2, d_model), layer)
        self.xgate        = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ogate        = clone(SublayerConnectionv2(d_model, dropout), layer)

    def forward(self, x, mask, *objs):
        # objs = (obj_feats, None, obj_mask, matrix)
        obj_feats, _, obj_mask, matrix = objs
        # obj_feats: [B, O, objcnndim]
        o = self.trans_obj(obj_feats)                               # → [B, O, d_model]
        # mask dims: obj_mask is [B, 1, O], OK for attention
        matrix = matrix.unsqueeze(-1)                               # → [B, T, O, 1]
        matrix4obj = matrix.transpose(1, 2)                         # → [B, O, T, 1]
        batch, objn, xn = matrix4obj.size(0), matrix4obj.size(1), matrix4obj.size(2)

        for i in range(self.layer):
            # 1) textual self-attn
            newx = self.res4mes_x[i](x, self.mhatt_x[i](x, x, x, mask))
            # 2) visual self-attn
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, obj_mask))

            # 3) text → image gating
            newx_ep = newx.unsqueeze(2).expand(batch, xn, objn, x.size(-1))
            o_ep    = newo.unsqueeze(1).expand(batch, xn, objn, o.size(-1))
            x2o_g   = torch.sigmoid(self.mhatt_x2o[i](torch.cat([newx_ep, o_ep], -1)))
            x2o     = (x2o_g * matrix * o_ep).sum(2)

            # 4) image → text gating
            x_ep    = newx.unsqueeze(1).expand(batch, objn, xn, x.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch, objn, xn, o.size(-1))
            o2x_g   = torch.sigmoid(self.mhatt_o2x[i](torch.cat([x_ep, newo_ep], -1)))
            o2x     = (o2x_g * matrix4obj * x_ep).sum(2)

            # 5) fuse & FFN
            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)
            x    = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o    = self.res4ffn_o[i](newo, self.ffn_o[i](newo))

        return x, o


def transformer(args):
    d_model, d_hidden, n_heads = args.d_model, args.d_hidden, args.n_heads
    src_vocab, trg_vocab         = args.src_vocab, args.trg_vocab

    src_emb_pos = nn.Sequential(Embeddings(d_model, src_vocab),
                                PositionalEncoding(d_model, args.input_drop_ratio))
    tgt_emb_pos = nn.Sequential(Embeddings(d_model, trg_vocab),
                                PositionalEncoding(d_model, args.input_drop_ratio))

    encoder = GATEncoder(d_model, d_hidden, n_heads, args.enc_dp, args.n_enclayers)
    decoder = Decoder(DecoderLayer(d_model, n_heads, d_hidden, args.dec_dp), args.n_layers)
    generator = Generator(d_model, trg_vocab)

    model = EncoderDecoder(encoder, decoder, src_emb_pos, tgt_emb_pos, generator)
    if args.share_vocab:
        model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
    if args.share_embed:
        model.generator.proj.weight = model.tgt_embed[0].lut.weight
    return model


def print_params(model):
    print('total parameters:', sum(p.numel() for p in model.parameters()))


def train(args, train_iter, dev, src, tgt, checkpoint):
    model = transformer(args).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(model); print_params(model)

    best_bleu = 0.0
    best_iter = 0
    offset = 0

    srcpadid = src.vocab.stoi['<pad>']
    tgtpadid = tgt.vocab.stoi['<pad>']

    if checkpoint is not None:
        print('Loading checkpoint...')
        model.load_state_dict(checkpoint['model'])
        offset = checkpoint.get('iters', 0)
        best_bleu = checkpoint.get('bleu', 0.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.optimizer == 'Noam':
        adamopt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        opt = NoamOpt(args.d_model, args.lr, args.warmup, adamopt, args.grad_clip, args.delay)
    else:
        raise NotImplementedError

    if args.resume and checkpoint is not None:
        opt.optimizer.load_state_dict(checkpoint['optim'])
        opt.set_steps(offset)
        print(f'Resuming from iter {offset}, best BLEU {best_bleu}')

    criterion = nn.KLDivLoss(reduction='sum')

    # label smoothing
    smoothing = 0.1
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / (args.trg_vocab - 2)
    one_hot = torch.full((args.trg_vocab,), smoothing_value, device=device)
    one_hot[tgtpadid] = 0.0
    one_hot = one_hot.unsqueeze(0)

    allboxfeats = pickle.load(open(args.boxfeat[0], 'rb'))
    valboxfeats = pickle.load(open(args.boxfeat[1], 'rb'))
    boxprobs    = pickle.load(open(args.boxprobs,   'rb'))

    start_time = time.time()
    max_steps   = args.maximum_steps * args.delay
    topk = 5
    objdim = args.objdim
    thre = 0.0

    for iters, batch in enumerate(train_iter, start=1):
        global_step = iters + offset
        if global_step > max_steps:
            print('Reached maximum training steps.')
            break

        model.train()
        t1 = time.time()

        sources, source_masks = prepare_sources(batch.src, srcpadid, args.share_vocab)
        sources, source_masks = sources.to(device), source_masks.to(device)

        tgt_in, tgt_out, tgt_mask, n_tokens = prepare_targets(batch.trg, tgtpadid)
        tgt_in, tgt_out, tgt_mask = tgt_in.to(device), tgt_out.to(device), tgt_mask.to(device)

        imgs, aligns, regions_num = batch.extra_0
        batch_size, seq_len = sources.size(0), sources.size(1)
        max_regions = max(regions_num)

        # 统一在 GPU 上创建
        obj_feat = torch.zeros(batch_size, max_regions, topk, objdim, device=device)
        obj_mask = torch.zeros(batch_size, 1, max_regions * topk, device=device)
        matrix   = torch.zeros(batch_size, seq_len, max_regions * topk, device=device)

        for ib, img in enumerate(imgs):
            # 先取出该图的所有 box features 和 probs
            boxfeat      = torch.tensor(allboxfeats[img], device=device).view(-1, 5, objdim)
            img_boxprobs = torch.tensor(boxprobs[img], device=device).view(-1)

            # region 数量
            num_regions = boxfeat.size(0)
            flat_len = num_regions * topk

            ge_thre = (img_boxprobs >= thre).byte()
            ge_thre[::5] = 1  # 每 5 个至少保留一个

            # 填充 mask 和 features
            obj_mask[ib, 0, :flat_len] = ge_thre
            obj_feat[ib, :num_regions] = boxfeat[:, :topk]

            # 构建 word–object 关系矩阵
            for word_id, region_id in aligns[ib]:
                idxs = torch.arange(topk, device=device) + region_id * topk
                matrix[ib, word_id, idxs] = ge_thre[idxs].float()
        
        
        obj_feat = obj_feat.view(batch_size, -1, objdim)   # [B, max_regions*topk, objdim]

       
        # 前向计算 & loss
        outputs = model(sources, tgt_in, source_masks, tgt_mask,
                        obj_feat, None, obj_mask, matrix)

        # 构造平滑标签
        # truth_p = one_hot.repeat(tgt_out.size(0), tgt_out.size(1), 1)
        # truth_p = truth_p.masked_scatter_(
        #     (tgt_out == tgtpadid).unsqueeze(2),
        #     torch.zeros(1, device=device)
        # )

        # build the smoothed one-hot matrix
        truth_p = one_hot.repeat(tgt_out.size(0), tgt_out.size(1), 1)        # [B, L, V]
        truth_p.scatter_(2, tgt_out.unsqueeze(2), confidence)               # put 'confidence' on the true label
        
        # mask out <pad> rows — broadcasting works with masked_fill_
        pad_mask = tgt_out.eq(tgtpadid).unsqueeze(2)                         # [B, L, 1]
        truth_p.masked_fill_(pad_mask, 0.0)                                  # zero all probs where tgt is <pad>

        # truth_p.scatter_(2, tgt_out.unsqueeze(2), confidence)

        loss = criterion(outputs, truth_p) / n_tokens.float().to(device)
        loss = loss / args.delay
        loss.backward()
        opt.step()

        t2 = time.time()
        print(f'{t2}: Iter {global_step:>6} | loss {loss.item()*args.delay:.4f} | '
              f'dt {(t2-t1):.2f}s | lr {opt._rate:.2e}')

        # eval & save
        if global_step % (args.eval_every * args.delay) == 0:
            with torch.no_grad():
                val_score = valid_model(args, model, dev, src, tgt, valboxfeats, boxprobs)
            print(f'Validation @ iter {global_step}: BLEU {val_score:.2f} (best {best_bleu:.2f})')
            if val_score > best_bleu:
                best_bleu, best_iter = val_score, global_step
                ckpt = {
                    'model': model.state_dict(),
                    'optim': opt.optimizer.state_dict(),
                    'args': args,
                    'bleu': best_bleu,
                    'iters': best_iter
                }
                torch.save(ckpt, f'{args.model_path}/{args.model}.best.pt')
                print(f'  → Saved new best checkpoint at iter {best_iter}')

        if global_step % (args.save_every * args.delay) == 0:
            backup = {
                'model': model.state_dict(),
                'optim': opt.optimizer.state_dict(),
                'args': args,
                'bleu': best_bleu,
                'iters': global_step
            }
            torch.save(backup, f'{args.model_path}/{args.model}.{global_step}.backup.pt')
            print(f'  → Saved backup checkpoint at iter {global_step}')

    # 训练结束后再 eval
    with torch.no_grad():
        final_score = valid_model(args, model, dev, src, tgt, valboxfeats, boxprobs)
    print(f'Final validation BLEU: {final_score:.2f} (best {best_bleu:.2f})')

    elapsed = (time.time() - start_time) / 60
    print(f'Training done in {elapsed:.1f} minutes. Best BLEU {best_bleu:.2f} @ iter {best_iter}')


# ----------------- 验证/测试部分 -----------------
def valid_model(args, model, dev, src, tgt, allboxfeats, boxprobs, dev_metrics=None):
    model.eval()
    initid   = tgt.vocab.stoi['<init>']
    eosid    = tgt.vocab.stoi['<eos>']
    srcpadid = src.vocab.stoi['<pad>']
    f = open(args.writetrans, 'w', encoding='utf-8')
    dev.init_epoch()
    decoding_times = []

    for j, dev_batch in enumerate(dev):
        sources, source_masks = prepare_sources(dev_batch.src, srcpadid, args.share_vocab)
        sources, source_masks = sources.cuda(), source_masks.cuda()

        imgs, aligns, regions_num = dev_batch.extra_0
        topk, objdim = 5, args.objdim
        max_regions  = max(regions_num)

        # 准备 multimodal inputs（同 train 中做法）
        obj_feat = sources.new_zeros(sources.size(0), max_regions, topk, objdim).float().cuda()
        obj_mask = sources.new_zeros(sources.size(0), 1, max_regions * topk).float().cuda()
        matrix   = sources.new_zeros(sources.size(0), sources.size(1), max_regions * topk).float().cuda()

        for ib, img in enumerate(imgs):
            boxfeat      = torch.tensor(allboxfeats[img], device='cuda').view(-1, 5, objdim)
            img_boxprobs = torch.tensor(boxprobs[img],   device='cuda').view(-1)
            ge_thre      = (img_boxprobs >= 0.0).byte()
            ge_thre[::5] = 1

            num_regions = boxfeat.size(0)
            flat_len    = num_regions * topk
            obj_mask[ib, 0, :flat_len] = ge_thre
            obj_feat[ib, :num_regions] = boxfeat[:, :topk]

            for word_id, region_id in aligns[ib]:
                idxs = torch.arange(topk, device='cuda') + region_id * topk
                matrix[ib, word_id, idxs] = ge_thre[idxs].float()

        # 把 obj_feat 拉平成 [B, max_regions*topk, objdim]
        obj_feat = obj_feat.view(sources.size(0), -1, objdim)

        start_t = time.time()
        if args.beam_size == 1:
            translations_id = greedy(
                args, model,
                sources, source_masks,
                initid, eosid,
                obj_feat, None, obj_mask, matrix        # ← 新增
            )
        else:
            translations_id = beam_search(
                args, model,
                sources, source_masks,
                initid, eosid,
                obj_feat, None, obj_mask, matrix        # ← 新增
            )
        
        decoding_times.append(time.time() - start_t)

        translations = tgt.reverse(translations_id.detach(), unbpe=True)
        for trans in translations:
            print(trans, file=f)

    f.close()
    # ... 计算 BLEU 放这里 ...
    status, bleuinfo = subprocess.getstatusoutput(
        'perl scripts/multi-bleu.perl -lc {} < {}'.format(args.ref, args.writetrans))
    bleu = re.findall(r'BLEU = (.*?),', bleuinfo)

    if len(bleu) == 0:
        print('bleu', bleuinfo)
        return 0
    if decoding_times:       # 防止空列表
        avg_ms = 1000 * sum(decoding_times) / len(decoding_times)
        print(f'average decoding latency: {avg_ms:.1f} ms')

    return float(bleu[0])


def decode(args, testset, src, tgt, checkpoint):
    with torch.no_grad():
        model = transformer(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(checkpoint['model'])

        valboxfeats = pickle.load(open(args.boxfeat[0], 'rb'))
        boxprobs = pickle.load(open(args.boxprobs, 'rb'))

        score = valid_model(args, model, testset, src, tgt, valboxfeats, boxprobs)
        print('bleu', score)



# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
#         super(EncoderDecoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.generator = generator
#         self.loss_computer = None

#     def forward(self, src, tgt, src_mask, tgt_mask, *objs):
#         hx, ho = self.encode(src, src_mask, *objs)
#         dec_outputs = self.decode(hx, src_mask, tgt, ho, objs[-2], tgt_mask)
#         return self.generator(dec_outputs)

#     def encode(self, src, src_mask, *objs):
#         return self.encoder(self.src_embed(src), src_mask, *objs)

#     def decode(self, memory, src_mask, tgt, objmem, objmask, tgt_mask=None):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, objmem, objmask, tgt_mask)

#     def generate(self, dec_outputs):
#         return self.generator(dec_outputs)

#     def addposition(self, x):
#         return self.tgt_embed[1](x)



class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, *objs):
        # encode 会处理多模态融合
        memory, _ = self.encoder(self.src_embed(src), src_mask, *objs)
        # decoder 只需要融合后的 memory + 文本 mask
        dec_out   = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return self.generator(dec_out)

    def generate(self, dec_outputs):
        return self.generator(dec_outputs)

    def encode(self, src, src_mask, *objs):
        """
        返回 encoder 的原始输出。  
        GATEncoder 会返回 (hx, ho)，因此保持 tuple，不要截断。
        """
        return self.encoder(self.src_embed(src), src_mask, *objs)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        只把 hx 传给 decoder。（如果需要 ho，可自行改动）
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def addposition(self, x: torch.Tensor) -> torch.Tensor:
        """
        给输入张量加上 **目标侧** 的位置编码。
        x 的形状 (B, T, H)；只用 tgt_embed 里的 PositionalEncoding。
        """
        return self.tgt_embed[1](x)   # self.tgt_embed = [Embeddings, PositionalEncoding]
