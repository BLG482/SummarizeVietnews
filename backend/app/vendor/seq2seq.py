"""Vietnamese abstractive summarization: BiGRU + Luong attention, from scratch.

Commands: prepare, train, evaluate, predict. See README.md / the notebook.
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from rouge_score import rouge_scorer
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def normalize(text):
    text = unicodedata.normalize('NFC', str(text)).replace('_', ' ')
    # Identical processing for pre-segmented CSV and ordinary inference text.
    def space_punctuation(match):
        start, end = match.span()
        char = match.group(0)
        prev_char = text[start - 1] if start else ''
        next_char = text[end] if end < len(text) else ''
        # Preserve numeric separators such as 1.500, 3,5 and 10:30.
        if prev_char.isdigit() and next_char.isdigit():
            return char
        return f' {char} '

    text = re.sub(r'[^\w\s]', space_punctuation, text, flags=re.UNICODE)
    return ' '.join(text.split())


def display_text(text):
    # Trim spaces before punctuation, but never join digits across punctuation:
    # "2025 . 15" must display as "2025. 15", not the different number "2025.15".
    text = re.sub(r'\s+([,.;:!?%\)\]\}])', r'\1', text)
    return re.sub(r'([\(\[\{])\s+', r'\1', text)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')



def assert_prepared_current(prepared):
    """Reject stale preprocessing outputs before either training stage."""
    prepared = Path(prepared)
    manifest_path = prepared / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'PREPARED thiếu {manifest_path.name}; hãy chạy lại prepare vào thư mục mới.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = sha(inspect.getsource(normalize))
    actual = manifest.get('normalize_source_sha256')
    if actual != expected:
        raise ValueError(
            'PREPARED được tạo bằng normalize khác hoặc thiếu hash kiểm tra. '
            f'Đổi tên PREPARED sang thư mục mới rồi chạy lại prepare. '
            f'expected={expected}, found={actual!r}'
        )
    train_path = prepared / 'train.jsonl'
    if not train_path.is_file():
        raise FileNotFoundError(f'PREPARED thiếu {train_path.name}; hãy chạy lại prepare vào thư mục mới.')
    with train_path.open(encoding='utf-8') as f:
        first_line = f.readline()
    try:
        row = json.loads(first_line)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('Dòng đầu train.jsonl không phải JSON hợp lệ; hãy đổi tên PREPARED và chạy lại prepare.') from exc
    missing = [key for key in ('lead_source', 'title') if not str(row.get(key, '')).strip()]
    if missing:
        raise ValueError(
            f'Dòng đầu train.jsonl thiếu trường {missing}. '
            'Đổi tên PREPARED sang thư mục mới rồi chạy lại prepare.'
        )
    return manifest

def prepare(args):
    out = Path(args.prepared)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Prepared folder must be empty. Use a new folder to avoid mixing runs.')
    csv.field_size_limit(10_000_000)
    seen = set()
    report = {'normalization': 'NFC, underscores to spaces, Unicode punctuation spacing; preserve punctuation between adjacent digits',
              'policy': 'title + article -> original summary; common CSV rows and source_row preserved; source normalized at model encoding only',
              'vocab_requested': args.vocab_size, 'splits': {},
              'normalize_source_sha256': sha(inspect.getsource(normalize))}
    for split in ('train', 'val', 'test'):
        stats = dict(read=0, kept=0, empty=0, duplicate_source=0)
        source = Path(args.data_dir) / f'{split}.csv'
        with source.open(encoding='utf-8-sig', newline='') as f, (out / f'{split}.jsonl').open('w', encoding='utf-8') as dest:
            reader = csv.DictReader(f)
            if not {'title', 'article', 'summary'}.issubset(reader.fieldnames or []):
                raise ValueError(f'{source}: expected title, article, and summary columns')
            for record, row in enumerate(tqdm(reader, desc=f'Clean {split}'), 1):
                stats['read'] += 1
                if None in row or row.get('title') is None or row.get('article') is None or row.get('summary') is None:
                    raise ValueError(f'{source}, CSV record {record}: malformed row')
                article = str(row['article'])
                summary = str(row['summary'])
                if not str(row['title']).strip() or not article.strip() or not summary.strip():
                    raise ValueError(f'{source}, record {record}: shared split unexpectedly contains an empty article/summary')
                key = sha(normalize(article))
                if key in seen:
                    raise ValueError(f'{source}, record {record}: duplicate article found; do not silently drop rows from the shared split')
                seen.add(key)
                source_row = int(row['source_row']) if row.get('source_row') not in (None, '') else record-1
                lead_source = str(row.get('lead_source') or '')
                if not lead_source.strip():
                    raise ValueError(f'{source}, record {record}: missing raw article for Lead-1')
                dest.write(json.dumps(dict(record=source_row, source_row=source_row, title=str(row['title']),
                                           article=article, summary=summary, lead_source=lead_source), ensure_ascii=False)+'\n')
                stats['kept'] += 1
        if not stats['kept']:
            raise ValueError(f'{split}: no usable records')
        report['splits'][split] = stats
        print(split, stats)

    def training_text():
        # Only training text. Short chunks prevent long documents being skipped by SentencePiece.
        for row in read_jsonl(out / 'train.jsonl'):
            for col in ('article', 'summary'):
                words = normalize(row[col]).split()
                for start in range(0, len(words), 200):
                    yield ' '.join(words[start:start+200])

    spm.SentencePieceTrainer.train(
        sentence_iterator=training_text(), model_prefix=str(out / 'tokenizer'),
        vocab_size=args.vocab_size, model_type='bpe', character_coverage=.9995,
        byte_fallback=True, normalization_rule_name='identity',
        pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=UNK,
        hard_vocab_limit=False, num_threads=2, max_sentence_length=32768,
        input_sentence_size=200000, shuffle_input_sentence=True,
    )
    report['tokenizer_sha256'] = hashlib.sha256((out / 'tokenizer.model').read_bytes()).hexdigest()
    save_json(out / 'manifest.json', report)
    print('Prepared:', out)


class EncodedDataset(Dataset):
    def __init__(self, path, tokenizer, max_source, max_target, limit=0, seed=42):
        self.rows = []
        stats = dict(rows=0, source_truncated=0, target_truncated=0, unknown=0, tokens=0)
        # Reservoir sampling gives a reproducible subset instead of only early CSV rows.
        raw = None
        if limit:
            raw, rng = [], random.Random(seed)
            for i, row in enumerate(read_jsonl(path)):
                if i < limit:
                    raw.append(row)
                else:
                    j = rng.randrange(i+1)
                    if j < limit:
                        raw[j] = row
        for row in tqdm(raw if raw is not None else read_jsonl(path), desc=f'Encode {Path(path).stem}'):
            src = tokenizer.encode(normalize(row['article']), out_type=int)
            tgt = tokenizer.encode(normalize(row['summary']), out_type=int)
            stats['rows'] += 1
            stats['source_truncated'] += len(src) > max_source-1
            stats['target_truncated'] += len(tgt) > max_target-2
            stats['unknown'] += src.count(UNK) + tgt.count(UNK)
            stats['tokens'] += len(src)+len(tgt)
            src = src[:max_source-1] + [EOS]
            tgt = [BOS] + tgt[:max_target-2] + [EOS]
            self.rows.append((np.asarray(src, dtype=np.int32), np.asarray(tgt, dtype=np.int32)))
        self.stats = stats
        if not self.rows:
            raise ValueError(f'Empty dataset: {path}')
        print(stats)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    source, target = zip(*rows)
    lengths = torch.tensor([len(x) for x in source], dtype=torch.long)
    source = pad_sequence([torch.tensor(x, dtype=torch.long) for x in source], batch_first=True, padding_value=PAD)
    target = pad_sequence([torch.tensor(x, dtype=torch.long) for x in target], batch_first=True, padding_value=PAD)
    return source, lengths, target


@dataclass
class Config:
    vocab_size: int
    embedding: int = 256
    hidden: int = 256
    layers: int = 1
    dropout: float = .2
    max_source: int = 768
    max_target: int = 128


class Seq2Seq(nn.Module):
    """Global general Luong attention with 1/sqrt(d_h) scaling and pre-tanh normalization.

    Teacher-forced decoder GRU is vectorized across target positions.
    Incremental inference uses exactly the same recurrent/attention equations.
    """
    def __init__(self, cfg, init_weights=True):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding, padding_idx=PAD)
        self.dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.GRU(cfg.embedding, cfg.hidden, cfg.layers, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout if cfg.layers > 1 else 0.)
        self.bridge = nn.Linear(2*cfg.hidden, cfg.hidden)
        self.decoder = nn.GRU(cfg.embedding, cfg.hidden, cfg.layers, batch_first=True,
                              dropout=cfg.dropout if cfg.layers > 1 else 0.)
        self.key = nn.Linear(2*cfg.hidden, cfg.hidden, bias=False)
        self.fuse = nn.Linear(3*cfg.hidden, cfg.embedding)
        self.output = nn.Linear(cfg.embedding, cfg.vocab_size)
        if init_weights:
            self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.cfg.embedding ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()
        for layer in (self.bridge, self.key, self.fuse, self.output):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, src, lengths):
        packed = pack_padded_sequence(self.dropout(self.embedding(src)), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_memory, h = self.encoder(packed)
        memory, _ = pad_packed_sequence(packed_memory, batch_first=True, total_length=src.size(1))
        h = h.reshape(self.cfg.layers, 2, src.size(0), self.cfg.hidden)
        bridged = self.bridge(torch.cat((h[:, 0], h[:, 1]), dim=-1))
        state = torch.tanh(F.layer_norm(bridged, (self.cfg.hidden,)))
        return memory, self.key(memory), src.eq(PAD), state

    def decode(self, tokens, state, memory, keys, padding):
        query, state = self.decoder(self.dropout(self.embedding(tokens)), state)
        scores = torch.bmm(query, keys.transpose(1, 2)) * (self.cfg.hidden ** -0.5)
        scores = scores.masked_fill(padding.unsqueeze(1), float('-inf'))
        attention = F.softmax(scores.float(), dim=-1).to(query.dtype)
        context = torch.bmm(attention, memory)
        fused_raw = self.fuse(torch.cat((query, context), dim=-1))
        fused = torch.tanh(F.layer_norm(fused_raw, (self.cfg.embedding,)))
        return self.output(self.dropout(fused)), state, attention

    def forward(self, src, lengths, decoder_input):
        memory, keys, padding, state = self.encode(src, lengths)
        return self.decode(decoder_input, state, memory, keys, padding)[0]


def banned_tokens(tokens, n):
    if n <= 0 or len(tokens) < n-1:
        return []
    prefix = tuple(tokens[-(n-1):]) if n > 1 else ()
    return list({tokens[i+n-1] for i in range(len(tokens)-n+1)
                 if tuple(tokens[i:i+n-1]) == prefix})


@torch.inference_mode()
def generate(model, tokenizer, text, beam_size=4, min_new=8, no_repeat=4, alpha=.6, max_new=None):
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(normalize(text), out_type=int)
    input_tokens_before_truncation = len(ids) + 1  # Include the EOS token appended by the model pipeline.
    truncated = len(ids) > model.cfg.max_source-1
    ids = ids[:model.cfg.max_source-1] + [EOS]
    src = torch.tensor([ids], device=device)
    memory, keys, padding, initial_state = model.encode(src, torch.tensor([len(ids)]))
    beams = [([BOS], 0., initial_state)]

    def rank(item):
        return item[1] / (((5+len(item[0])-1)/6)**alpha)

    max_steps = model.cfg.max_target-1 if max_new is None else max(1, min(int(max_new), model.cfg.max_target-1))
    # Leave the final available step able to emit EOS; otherwise a tight
    # max_new budget can make a well-formed ending impossible.
    min_new = min(min_new, max_steps-1)
    for step in range(max_steps):
        active = [(index, beam) for index, beam in enumerate(beams) if beam[0][-1] != EOS]
        candidates = [(sequence, score, hidden) for sequence, score, hidden in beams if sequence[-1] == EOS]
        if active:
            active_count = len(active)
            tokens = torch.tensor([[beam[0][-1]] for _, beam in active], device=device)
            states = torch.cat([beam[2] for _, beam in active], dim=1)
            expanded_memory = memory.expand(active_count, -1, -1)
            expanded_padding = padding.expand(active_count, -1)
            if isinstance(keys, tuple):
                expanded_keys = tuple(value.expand(active_count, *value.shape[1:]) for value in keys)
            else:
                expanded_keys = keys.expand(active_count, *keys.shape[1:])
            logits, next_states, _ = model.decode(
                tokens, states, expanded_memory, expanded_keys, expanded_padding
            )
            log_probs = F.log_softmax(logits[:, -1].float(), dim=-1)
            for active_index, (_, (sequence, score, hidden)) in enumerate(active):
                probs = log_probs[active_index].clone()
                probs[[PAD, BOS, UNK]] = float('-inf')
                if step < min_new:
                    probs[EOS] = float('-inf')
                blocked = banned_tokens(sequence[1:], no_repeat)
                if blocked:
                    probs[blocked] = float('-inf')
                values, indices = probs.topk(min(beam_size, probs.numel()))
                next_hidden = next_states[:, active_index:active_index+1]
                for value, index in zip(values.tolist(), indices.tolist()):
                    if math.isfinite(value):
                        candidates.append((sequence+[index], score+value, next_hidden))
        if not candidates:
            break
        # Python's stable sort preserves the old parent-beam/top-k order on score ties.
        beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
        if all(item[0][-1] == EOS for item in beams):
            break
    # Do not force an inferior early EOS when a stronger unfinished beam remains.
    sequence = max(beams, key=rank)[0]
    generated = [x for x in sequence[1:] if x not in (PAD, BOS, EOS)]
    return dict(summary=display_text(tokenizer.decode(generated)), source_truncated=truncated,
                input_truncated=truncated, input_tokens_before_truncation=input_tokens_before_truncation,
                ended_with_eos=sequence[-1] == EOS, generated_subwords=len(generated))

class VietnameseRougeTokenizer:
    def tokenize(self, text):
        # The package's default [a-z0-9] tokenizer discards Vietnamese accents.
        # Here scores use Unicode syllable tokens, NOT Vietnamese word segmentation.
        text = unicodedata.normalize('NFC', str(text)).casefold().replace('_', ' ')
        return re.findall(r'[^\W_]+', text, flags=re.UNICODE)


def sampled_rows(path, limit, seed=42):
    reservoir, rng = [], random.Random(seed)
    for i, row in enumerate(read_jsonl(path)):
        if not limit or i < limit:
            reservoir.append(row)
        else:
            j = rng.randrange(i+1)
            if j < limit:
                reservoir[j] = row
    return reservoir


def _source_fields(row):
    article_field = str(row.get('article', ''))
    title = str(row.get('title') or '')
    lead_source = row.get('lead_source')
    match = re.search(r'\bTiêu đề:\s*(.*?)\s+Nội dung:\s*(.*)$',
                      article_field, flags=re.IGNORECASE | re.DOTALL)
    if match:
        if not title.strip():
            title = match.group(1).strip()
        if lead_source is None or not str(lead_source).strip():
            lead_source = match.group(2).strip()
    if lead_source is None or not str(lead_source).strip():
        lead_source = article_field
    return title, str(lead_source)


def _has_numbers_not_in_source(text, source_numbers):
    return bool(set(re.findall(r'\d+', str(text))) - source_numbers)


def evaluate_rows(model, tokenizer, rows, beam_size=1):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                     use_stemmer=False, tokenizer=VietnameseRougeTokenizer())
    names = ('rouge1', 'rouge2', 'rougeL')
    totals = {name: 0.0 for name in names}
    lead_totals = {name: 0.0 for name in names}
    title_totals = {name: 0.0 for name in names}
    model_number_mismatches = 0
    reference_number_mismatches = 0
    predictions = []
    for row in tqdm(rows, desc='Generate + ROUGE'):
        result = generate(model, tokenizer, row['article'], beam_size=beam_size)
        title, lead_source = _source_fields(row)
        lead = re.split(r'(?<=[.!?])\s+', lead_source.strip(), maxsplit=1)[0]
        reference = row['summary']
        scores = scorer.score(reference, result['summary'])
        lead_scores = scorer.score(reference, lead)
        title_scores = scorer.score(reference, title)
        source_numbers = set(re.findall(r'\d+', title + ' ' + lead_source))
        model_numbers_missing = _has_numbers_not_in_source(result['summary'], source_numbers)
        reference_numbers_missing = _has_numbers_not_in_source(reference, source_numbers)
        model_number_mismatches += model_numbers_missing
        reference_number_mismatches += reference_numbers_missing
        per_row = {}
        for key in names:
            totals[key] += scores[key].fmeasure
            lead_totals[key] += lead_scores[key].fmeasure
            title_totals[key] += title_scores[key].fmeasure
            per_row[key] = 100.0 * float(scores[key].fmeasure)
        source_row = int(row.get('source_row', row['record']))
        predictions.append(dict(record=source_row, source_row=source_row,
                                title=title, reference=reference, reference_summary=reference,
                                generated_summary=result['summary'], rouge1=per_row['rouge1'],
                                rouge2=per_row['rouge2'], rougeL=per_row['rougeL'],
                                lead1=lead,
                                generated_numbers_not_in_source=model_numbers_missing,
                                reference_numbers_not_in_source=reference_numbers_missing,
                                **result))
    n = len(rows)
    if not n:
        raise ValueError('Evaluation set is empty')
    raw_means = {key: totals[key] / n for key in names}
    truncated = sum(bool(p['input_truncated']) for p in predictions)
    # Automatic digit-presence diagnostic only; it is not a human factuality judgment.
    metrics = dict(n=n,
                   tokenizer='NFC + casefold + underscore-to-space + Unicode [^\\W_]+; no stemming',
                   beam_size=beam_size,
                   model_f1_raw=raw_means,
                   model={key: round(100.0 * raw_means[key], 4) for key in names},
                   lead1={key: round(100.0 * lead_totals[key] / n, 4) for key in names},
                   title_baseline={key: round(100.0 * title_totals[key] / n, 4) for key in names},
                   numbers_not_in_source_percent={
                       'model': 100.0 * model_number_mismatches / n,
                       'reference': 100.0 * reference_number_mismatches / n,
                   },
                   input_truncated_count=truncated,
                   input_truncated_rate_percent=100.0 * truncated / n,
                   source_truncated=truncated,
                   without_eos=sum(not p['ended_with_eos'] for p in predictions))
    return metrics, predictions

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def _lazy_rebuild_tensor(storage, storage_offset, size, stride, *args):
    return ('_TENSOR_', storage, int(storage_offset), tuple(size), tuple(stride))


class _LazyZipUnpickler( __import__('pickle').Unpickler ):
    def persistent_load(self, saved_id):
        if isinstance(saved_id, tuple) and saved_id and saved_id[0] == 'storage':
            _, storage_type, key, _location, numel = saved_id[:5]
            return ('_STORAGE_', storage_type, str(key), int(numel))
        return saved_id

    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_tensor_v3'):
            return _lazy_rebuild_tensor
        return super().find_class(module, name)


def load_model(run, device):
    import gc
    import io
    import zipfile
    run = Path(run)
    ckpt_file = run / 'best.pt'
    print(f'[Seq2Seq] Đang đọc {ckpt_file} (chế độ tiết kiệm RAM, bỏ qua optimizer 1.8GB)...', flush=True)
    with zipfile.ZipFile(ckpt_file, 'r') as zf:
        names = zf.namelist()
        pkl_entry = next(n for n in names if n.endswith('/data.pkl') or n == 'data.pkl')
        prefix = pkl_entry[:-len('data.pkl')]
        checkpoint = _LazyZipUnpickler(io.BytesIO(zf.read(pkl_entry))).load()
        tokenizer_path = run / 'tokenizer.model'
        if sha_bytes(tokenizer_path) != checkpoint['tokenizer_sha256']:
            raise ValueError('Checkpoint/tokenizer mismatch')
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        cfg = Config(**checkpoint['config'])
        epoch = checkpoint.get('epoch')
        lazy_model_state = checkpoint['model']
        checkpoint.clear()
        del checkpoint
        gc.collect()
        model = Seq2Seq(cfg, init_weights=False)
        params = dict(model.named_parameters())
        params.update(dict(model.named_buffers()))
        with torch.no_grad():
            for name, target_tensor in params.items():
                info = lazy_model_state[name]
                _, storage_info, storage_offset, size, stride = info
                _, _storage_type, key, _numel = storage_info
                raw_bytes = zf.read(f'{prefix}data/{key}')
                flat = torch.frombuffer(bytearray(raw_bytes), dtype=target_tensor.dtype)
                view = torch.as_strided(flat, size, stride, storage_offset)
                target_tensor.copy_(view)
                del raw_bytes, flat, view
        lazy_model_state.clear()
        del lazy_model_state
    gc.collect()
    model = model.to(device)
    print(f'[Seq2Seq] Đã nạp xong (epoch {epoch}, device={device}).', flush=True)
    return model.eval(), tokenizer, {'epoch': epoch, 'config': asdict(cfg)}


def sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    seed_everything(args.seed)
    device, prepared, run = device_for(args.device), Path(args.prepared), Path(args.run)
    assert_prepared_current(prepared)
    run.mkdir(parents=True, exist_ok=True)
    if not args.resume and (run / 'last.pt').exists():
        raise ValueError('Run already exists. Pass --resume or choose a new --run.')
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer_path = prepared / 'tokenizer.model'
    tokenizer_hash = sha_bytes(tokenizer_path)
    if tokenizer_hash != manifest['tokenizer_sha256']:
        raise ValueError('Prepared tokenizer was changed. Re-prepare in a new folder.')
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    cfg = Config(tokenizer.get_piece_size(), args.embedding, args.hidden, args.layers,
                 args.dropout, args.max_source, args.max_target)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(run / 'last.pt', map_location=device, weights_only=True)
        if checkpoint['config'] != asdict(cfg) or checkpoint['tokenizer_sha256'] != tokenizer_hash:
            raise ValueError('Resume configuration/tokenizer differs from the saved run')
        for name in ('seed', 'max_train', 'max_val', 'batch_size', 'lr', 'eval_samples', 'eval_beam'):
            if checkpoint['args'][name] != vars(args)[name]:
                raise ValueError(f'Resume must keep {name}={checkpoint["args"][name]}')
    train_set = EncodedDataset(prepared/'train.jsonl', tokenizer, cfg.max_source, cfg.max_target, args.max_train, args.seed)
    val_set = EncodedDataset(prepared/'val.jsonl', tokenizer, cfg.max_source, cfg.max_target, args.max_val, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    val_examples = sampled_rows(prepared/'val.jsonl', args.eval_samples, args.seed)
    model = Seq2Seq(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=1)
    best, bad_epochs, start, history = -1., 0, 1, []
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best, bad_epochs = checkpoint['best'], checkpoint['bad_epochs']
        start, history = checkpoint['epoch']+1, checkpoint['history']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        random.setstate(checkpoint['random_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    (run/'tokenizer.model').write_bytes(tokenizer_path.read_bytes())
    save_json(run/'config.json', asdict(cfg))
    save_json(run/'data_manifest.json', manifest)
    save_json(run/'encoding_stats.json', dict(train=train_set.stats, val=val_set.stats))
    print('Device:', device, 'Parameters:', sum(p.numel() for p in model.parameters()))
    print('100% teacher forcing for training CE; validation ROUGE uses autoregressive generation.')
    for epoch in range(start, args.epochs+1):
        model.train()
        train_loss, train_tokens = 0., 0
        bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for src, lengths, target in bar:
            src, target = src.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Shift exactly once: BOS,... -> ...,EOS. PAD never contributes to CE.
            logits = model(src, lengths, target[:, :-1])
            gold = target[:, 1:]
            total = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), gold.reshape(-1),
                                    ignore_index=PAD, reduction='sum')
            tokens = int(gold.ne(PAD).sum())
            loss = total/tokens
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss. Inspect the batch and learning rate.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_loss += total.item()
            train_tokens += tokens
            bar.set_postfix(ce=round(train_loss/train_tokens, 4))
        model.eval()
        val_loss, val_tokens = 0., 0
        with torch.inference_mode():
            for src, lengths, target in tqdm(val_loader, desc='Validation CE'):
                src, target = src.to(device), target.to(device)
                logits = model(src, lengths, target[:, :-1])
                gold = target[:, 1:]
                val_loss += F.cross_entropy(logits.reshape(-1, cfg.vocab_size), gold.reshape(-1), ignore_index=PAD, reduction='sum').item()
                val_tokens += int(gold.ne(PAD).sum())
        metrics, predictions = evaluate_rows(model, tokenizer, val_examples, beam_size=args.eval_beam)
        current = metrics['model_f1_raw']['rougeL']
        scheduler.step(current)
        improved = current > best
        if improved:
            best, bad_epochs = current, 0
        else:
            bad_epochs += 1
        row = dict(epoch=epoch, train_ce=train_loss/train_tokens, val_ce=val_loss/val_tokens,
                   selection_rougeL_f1_raw=current, lr=optimizer.param_groups[0]['lr'], validation=metrics)
        history.append(row)
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), config=asdict(cfg), args=vars(args),
                     tokenizer_sha256=tokenizer_hash, best=best, bad_epochs=bad_epochs,
                     history=history, torch_rng=torch.get_rng_state(), random_rng=random.getstate(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        atomic_save(run/'last.pt', state)
        if improved:
            atomic_save(run/'best.pt', state)
            save_json(run/'best_val_predictions.json', predictions)
        save_json(run/'history.json', history)
        print(json.dumps(row, ensure_ascii=False))
        for item in predictions[:2]:
            print('REF:', display_text(item['reference']), '\nPRED:', item['summary'])
        if bad_epochs >= args.patience:
            print('Early stopping on validation generation ROUGE-L.')
            break


def evaluate(args):
    run = Path(args.run)
    model, tokenizer, checkpoint = load_model(run, device_for(args.device))
    prepared = Path(args.prepared)
    if sha_bytes(prepared/'tokenizer.model') != checkpoint['tokenizer_sha256']:
        raise ValueError('Evaluation prepared data/tokenizer differs from the training run')
    rows = sampled_rows(prepared/f'{args.split}.jsonl', args.limit)
    metrics, predictions = evaluate_rows(model, tokenizer, rows, args.beam_size)
    metrics['checkpoint_epoch'] = checkpoint['epoch']
    metrics['split'] = args.split
    metrics['limit_requested'] = args.limit
    tag = f'{args.split}_{len(rows)}_beam{args.beam_size}'
    save_json(run/f'{tag}_metrics.json', metrics)
    with (run/f'{tag}_predictions.jsonl').open('w', encoding='utf-8') as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--prepared', default='prepared')
    p.add_argument('--vocab-size', type=int, default=12000)
    p.set_defaults(func=prepare)
    p = commands.add_parser('train')
    p.add_argument('--prepared', default='prepared')
    p.add_argument('--run', default='runs/bigru')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--embedding', type=int, default=256)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--max-source', type=int, default=768)
    p.add_argument('--max-target', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--eval-samples', type=int, default=100)
    p.add_argument('--eval-beam', type=int, default=4)
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--max-val', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.set_defaults(func=train)
    p = commands.add_parser('evaluate')
    p.add_argument('--prepared', default='prepared')
    p.add_argument('--run', default='runs/bigru')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--limit', type=int, default=0, help='0 = entire split; positive = seeded sample')
    p.add_argument('--beam-size', type=int, default=4)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.set_defaults(func=evaluate)
    p = commands.add_parser('predict')
    p.add_argument('--run', default='runs/bigru')
    p.add_argument('--text', required=True)
    p.add_argument('--beam-size', type=int, default=4)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    def predict(args):
        model, tokenizer, _ = load_model(args.run, device_for(args.device))
        print(json.dumps(generate(model, tokenizer, args.text, args.beam_size), ensure_ascii=False, indent=2))
    p.set_defaults(func=predict)
    args = parser.parse_args()
    for key in ('batch_size', 'epochs', 'embedding', 'hidden', 'layers', 'patience', 'eval_samples', 'beam_size'):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    for key in ('limit', 'max_train', 'max_val'):
        if hasattr(args, key) and getattr(args, key) < 0:
            parser.error(f'{key} must be nonnegative')
    if hasattr(args, 'max_source') and (args.max_source < 2 or args.max_target < 12):
        parser.error('max-source >= 2 and max-target >= 12 required')
    args_dict = vars(args)
    func = args_dict.pop('func')  # Keep checkpoint arguments weights_only-compatible.
    func(args)


if __name__ == '__main__':
    main()
