(() => {
  'use strict';

  const MODELS = {
    seq2seq: { label: 'Seq2Seq', sub: 'BiGRU + attention', cls: 'm-seq' },
    vit5: { label: 'ViT5', sub: 'Transformer', cls: 'm-vit' },
  };
  const ORDER = ['seq2seq', 'vit5'];
  const STATE_TEXT = { ready: 'sẵn sàng', loading: 'đang nạp', not_loaded: 'chưa nạp', error: 'lỗi' };
  const PAGE = 15;
  const view = document.getElementById('view');

  const state = {
    current: null, // lượt tóm tắt đang xem
    blind: false,
    order: ORDER.slice(), // thứ tự hiển thị A/B khi đánh giá mù
    form: { title: '', article: '', reference: '', sample_id: null },
    samples: { q: '', offset: 0 },
    history: { offset: 0 },
  };

  // ---------- tiện ích ----------
  function el(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === false || value == null) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(kid));
    }
    return node;
  }

  const API_BASE = '/api';

  async function api(path, options = {}) {
    const response = await fetch(API_BASE + path, {
      method: options.method || 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
    if (!response.ok) {
      let message = response.statusText || ('HTTP ' + response.status);
      try {
        const data = await response.json();
        message = typeof data.detail === 'string' ? data.detail
          : response.status === 422 ? 'Dữ liệu gửi lên chưa hợp lệ.' : JSON.stringify(data.detail);
      } catch (_) { /* giữ statusText */ }
      throw new Error(message);
    }
    return response.json();
  }

  const nf = (v, digits = 2) => (v == null ? '–'
    : v.toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits }));
  const secs = (ms) => (ms == null ? '–' : nf(ms / 1000, 2) + ' giây');
  const when = (iso) => new Date(iso).toLocaleString('vi-VN',
    { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' });
  const words = (text) => (text.trim().match(/\S+/g) || []).length;
  const showMsg = (text) => { const m = document.getElementById('msg'); if (m) m.textContent = text; };

  // ---------- trạng thái mô hình ----------
  let statusTimer = null;
  async function refreshStatus() {
    clearTimeout(statusTimer);
    let models = [];
    try { models = await api('/models'); } catch (_) { /* server chưa sẵn sàng */ }
    document.getElementById('status').replaceChildren(...models.map((m) => {
      const name = (MODELS[m.key] || {}).label || m.label;
      const chip = el('li', {
        class: 'chip st-' + m.state,
        title: m.state === 'error' ? m.error + '\nBấm để nạp lại.' : m.description,
      }, el('span', { class: 'dot' }), name, el('small', { text: STATE_TEXT[m.state] || m.state }));
      if (m.state === 'error') {
        chip.addEventListener('click', async () => { await api(`/models/${m.key}/reload`, { method: 'POST' }); refreshStatus(); });
      }
      return chip;
    }));
    if (models.some((m) => m.state === 'loading' || m.state === 'not_loaded')) {
      statusTimer = setTimeout(refreshStatus, 3000);
    }
  }

  // ---------- hiển thị một lượt tóm tắt ----------
  const WORD = /[\p{L}\p{N}]+/gu;
  const tokens = (text) => text.normalize('NFC').toLowerCase().replace(/_/g, ' ').match(WORD) || [];

  function summaryNode(summary, source) {
    const known = new Set(tokens(source));
    const p = el('p', { class: 'summary' });
    for (const part of summary.split(/(\s+)/)) {
      if (!part) continue;
      const ts = tokens(part);
      if (ts.length && !ts.some((t) => known.has(t))) p.append(el('span', { class: 'novel', text: part }));
      else p.append(document.createTextNode(part));
    }
    return p;
  }

  function sizeComparisonBar(r, sub) {
    const srcText = (sub.article || '').trim();
    const sumText = (r.summary || '').trim();
    const srcWords = sub.n_words || words(srcText);
    const srcChars = srcText.length;
    const sumWords = words(sumText);
    const sumChars = sumText.length;
    const ratio = srcWords > 0 ? Math.round((sumWords / srcWords) * 100) : 0;
    return el('div', { class: 'size-compare' },
      el('div', { class: 'size-box' },
        el('span', { class: 'size-label', text: 'Kích thước văn bản nguồn' }),
        el('strong', { text: `${srcWords.toLocaleString('vi-VN')} từ` }),
        el('small', { text: ` (${srcChars.toLocaleString('vi-VN')} ký tự)` })),
      el('span', { class: 'size-arrow', text: '→' }),
      el('div', { class: 'size-box' },
        el('span', { class: 'size-label', text: 'Kích thước bản tóm tắt' }),
        el('strong', { text: `${sumWords.toLocaleString('vi-VN')} từ` }),
        el('small', { text: ` (${sumChars.toLocaleString('vi-VN')} ký tự · bằng ${ratio}% bản gốc)` })));
  }

  function facts(r, sub) {
    const srcText = (sub.article || '').trim();
    const sumText = (r.summary || '').trim();
    const srcWords = sub.n_words || words(srcText);
    const sumWords = words(sumText);
    const tokenText = r.input_tokens != null ? `${r.input_tokens.toLocaleString('vi-VN')} / 1.024 token` : 'tối đa 1.024 token';
    const rows = [
      ['Văn bản nguồn', `${srcWords.toLocaleString('vi-VN')} từ (${srcText.length.toLocaleString('vi-VN')} ký tự)`],
      ['Bản tóm tắt', `${sumWords.toLocaleString('vi-VN')} từ (${sumText.length.toLocaleString('vi-VN')} ký tự)`],
      ['Thời gian xử lý', secs(r.latency_ms)],
      ['Token đầu vào', `${tokenText} (${r.truncated ? 'bị cắt bớt' : 'đủ toàn văn'})`],
    ];
    if (r.rougeL != null) rows.push(['ROUGE-1', nf(r.rouge1)], ['ROUGE-2', nf(r.rouge2)], ['ROUGE-L', nf(r.rougeL)]);
    return el('dl', { class: 'facts' }, rows.map(([k, v]) => el('div', {}, el('dt', { text: k }), el('dd', { text: v }))));
  }

  function resultCard(r, sub, hiddenLabel) {
    const meta = MODELS[r.model_key];
    const card = el('article', { class: 'card ' + (hiddenLabel ? '' : meta.cls) });
    card.append(el('h2', {}, hiddenLabel || meta.label,
      hiddenLabel ? null : el('span', { class: 'sub', text: meta.sub })));
    if (r.error) {
      card.append(el('p', { class: 'error', text: 'Không tóm tắt được. ' + r.error }));
      return card;
    }
    card.append(sizeComparisonBar(r, sub));
    card.append(summaryNode(r.summary, sub.title + ' ' + sub.article));
    if (r.unsupported_numbers && r.unsupported_numbers.length) {
      card.append(el('p', { class: 'warn', text: 'Số không có trong bài gốc: ' + r.unsupported_numbers.join(', ') }));
    }
    if (!hiddenLabel) card.append(facts(r, sub)); // thời gian và việc bị cắt sẽ lộ tên mô hình nên ẩn khi chấm mù
    return card;
  }

  function voteBar(sub, order, blindNow) {
    if (sub.results.filter((r) => !r.error).length < 2) return el('div');
    const choices = blindNow
      ? [['Bản A tốt hơn', order[0]], ['Ngang nhau', 'tie'], ['Bản B tốt hơn', order[1]]]
      : [[MODELS.seq2seq.label + ' tốt hơn', 'seq2seq'], ['Ngang nhau', 'tie'], [MODELS.vit5.label + ' tốt hơn', 'vit5']];
    const picked = sub.vote && sub.vote.choice;
    const row = el('div', { class: 'vote-row' }, choices.map(([text, key]) => el('button', {
      type: 'button', class: 'ghost' + (picked === key ? ' on' : ''), text,
      'aria-pressed': picked === key ? 'true' : 'false',
      onclick: () => castVote(sub, key, sub.vote ? sub.vote.blind : blindNow),
    })));
    return el('div', { class: 'vote', role: 'group', 'aria-label': 'Chọn bản tóm tắt tốt hơn' },
      el('p', { class: 'vote-q', text: picked ? 'Đã ghi nhận lựa chọn của bạn. Bấm lại để đổi.' : 'Bản nào tóm tắt tốt hơn?' }), row);
  }

  async function castVote(sub, choice, blind) {
    try {
      state.current = await api(`/submissions/${sub.id}/vote`, { method: 'POST', body: { choice, blind } });
      renderResults();
    } catch (e) { showMsg('Không ghi được bình chọn: ' + e.message); }
  }

  function renderResults() {
    const host = document.getElementById('results');
    const sub = state.current;
    if (!host || !sub) return;
    const blindNow = state.blind && !sub.vote;
    const order = blindNow ? state.order : ORDER;
    const byKey = Object.fromEntries(sub.results.map((r) => [r.model_key, r]));
    const grid = el('div', { class: 'grid2' });
    order.forEach((key, i) => { if (byKey[key]) grid.append(resultCard(byKey[key], sub, blindNow ? 'Bản ' + 'AB'[i] : null)); });
    host.replaceChildren(grid,
      el('p', { class: 'legend' }, el('span', { class: 'novel', text: 'Gạch chấm' }), ' đánh dấu âm tiết không có trong bài gốc.'),
      voteBar(sub, order, blindNow));
  }

  // ---------- trang So sánh ----------
  function viewCompare() {
    const f = state.form;
    const title = el('input', { id: 'title', type: 'text', value: f.title, autocomplete: 'off', maxlength: '1000',
      placeholder: 'Tiêu đề bài báo (không bắt buộc, tối đa 1.000 ký tự)' });
    const article = el('textarea', { id: 'article', rows: '10', maxlength: '60000',
      placeholder: 'Dán nội dung bài báo vào đây (tối thiểu 30 ký tự, tối đa 60.000 ký tự)' });
    const reference = el('textarea', { id: 'reference', rows: '3', maxlength: '5000',
      placeholder: 'Dán bản tóm tắt chuẩn để chấm ROUGE (tối đa 5.000 ký tự)' });
    article.value = f.article;
    reference.value = f.reference;
    const count = el('span', { class: 'muted' });
    const limitHint = el('p', { class: 'limit-hint' });
    const updateCount = () => {
      const nWords = words(article.value);
      const nChars = article.value.trim().length;
      count.textContent = `(${nWords.toLocaleString('vi-VN')} từ · ${nChars.toLocaleString('vi-VN')} / 60.000 ký tự)`;
      if (nChars > 0 && nChars < 30) {
        limitHint.className = 'limit-hint limit-warn';
        limitHint.textContent = `Cần nhập tối thiểu 30 ký tự (hiện tại: ${nChars} ký tự).`;
      } else if (nWords > 780) {
        limitHint.className = 'limit-hint limit-warn';
        limitHint.textContent = `Lưu ý: Bài báo có ${nWords.toLocaleString('vi-VN')} từ — có thể vượt ngưỡng 1.024 token đầu vào của mô hình (≈ 700–800 từ), phần vượt quá sẽ tự động cắt bớt.`;
      } else {
        limitHint.className = 'limit-hint';
        limitHint.textContent = 'Ràng buộc văn bản nguồn: Tối thiểu 30 ký tự, tối đa 60.000 ký tự · Ngưỡng nhận diện tối đa của 2 mô hình: 1.024 token đầu vào (≈ 700–800 từ) và tối đa 256 token đầu ra.';
      }
    };
    updateCount();
    title.addEventListener('input', () => { f.title = title.value; });
    article.addEventListener('input', () => { f.article = article.value; updateCount(); });
    reference.addEventListener('input', () => { f.reference = reference.value; });

    const blind = el('input', { type: 'checkbox', id: 'blind' });
    blind.checked = state.blind;
    blind.addEventListener('change', () => { state.blind = blind.checked; renderResults(); });

    const run = el('button', { type: 'button', class: 'primary', text: 'Tóm tắt bằng cả hai mô hình' });
    const results = el('section', { id: 'results', 'aria-live': 'polite' });
    run.addEventListener('click', async () => {
      showMsg('');
      if (article.value.trim().length < 30) {
        showMsg('Hãy dán nội dung bài báo (ít nhất 30 ký tự) trước khi tóm tắt.');
        return;
      }
      run.disabled = true;
      run.textContent = 'Đang tóm tắt…';
      results.replaceChildren(el('p', { class: 'muted',
        text: 'Hai mô hình đang chạy. Lần đầu có thể lâu hơn vì mô hình cần nạp.' }));
      try {
        state.current = await api('/summarize', { method: 'POST', body: {
          title: title.value, article: article.value,
          reference: reference.value.trim() || null, sample_id: f.sample_id } });
        state.order = Math.random() < 0.5 ? ORDER.slice() : ORDER.slice().reverse();
        renderResults();
      } catch (e) {
        results.replaceChildren();
        showMsg('Không tóm tắt được: ' + e.message);
      } finally {
        run.disabled = false;
        run.textContent = 'Tóm tắt bằng cả hai mô hình';
        refreshStatus();
      }
    });

    view.replaceChildren(
      el('h1', { text: 'So sánh hai bản tóm tắt' }),
      el('div', { class: 'field' }, el('label', { for: 'title', text: 'Tiêu đề' }), title),
      el('div', { class: 'field' }, el('label', { for: 'article' }, 'Bài báo ', count), article, limitHint),
      el('details', { open: f.reference ? true : false },
        el('summary', { text: 'Thêm bản tóm tắt tham chiếu để chấm ROUGE' }), reference),
      el('div', { class: 'actions' },
        el('label', { class: 'check' }, blind, 'Đánh giá mù: ẩn tên mô hình cho đến khi bạn chọn'), run),
      el('p', { class: 'msg', id: 'msg', role: 'alert' }),
      results);
    renderResults();
  }

  // ---------- trang Bài mẫu ----------
  async function viewSamples() {
    const s = state.samples;
    const search = el('input', { type: 'search', value: s.q, placeholder: 'Tìm theo tiêu đề', 'aria-label': 'Tìm bài mẫu' });
    const list = el('div', { class: 'list' });
    const pager = el('div', { class: 'pager' });
    let timer = null;
    search.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => { s.q = search.value; s.offset = 0; load(); }, 300);
    });

    async function load() {
      const data = await api(`/samples?q=${encodeURIComponent(s.q)}&offset=${s.offset}&limit=${PAGE}`);
      if (!data.available) {
        list.replaceChildren(el('div', { class: 'empty' }, 'Chưa có bài mẫu. Đặt file CSV test (cột ',
          el('code', { text: 'title' }), ', ', el('code', { text: 'article' }), ', ', el('code', { text: 'summary' }),
          ') tại ', el('code', { text: data.path }), ' rồi tải lại trang.'));
        pager.replaceChildren();
        return;
      }
      list.replaceChildren(...data.items.map((it) => el('article', { class: 'row' },
        el('h3', { text: it.title || '(không có tiêu đề)' }),
        el('p', { text: it.preview + '…' }),
        el('button', { type: 'button', class: 'ghost', text: 'Dùng bài này', onclick: () => useSample(it.id) }))));
      if (!data.items.length) list.replaceChildren(el('div', { class: 'empty', text: 'Không có bài nào khớp từ khóa.' }));
      const end = Math.min(s.offset + PAGE, data.total);
      pager.replaceChildren(
        el('button', { type: 'button', class: 'ghost', text: 'Trước', disabled: s.offset === 0,
          onclick: () => { s.offset = Math.max(0, s.offset - PAGE); load(); } }),
        el('span', { text: data.total ? `${s.offset + 1}–${end} trên ${data.total}` : '0 bài' }),
        el('button', { type: 'button', class: 'ghost', text: 'Sau', disabled: end >= data.total,
          onclick: () => { s.offset += PAGE; load(); } }));
    }

    view.replaceChildren(el('h1', { text: 'Bài mẫu từ tập test' }), search, list, pager);
    await load();
  }

  async function useSample(id) {
    try {
      const sample = await api('/samples/' + id);
      state.form = { title: sample.title, article: sample.article, reference: sample.reference, sample_id: sample.id };
      state.current = null;
      goto('compare');
    } catch (e) { alert('Không mở được bài mẫu: ' + e.message); }
  }

  // ---------- trang Lịch sử ----------
  async function viewHistory() {
    const h = state.history;
    const data = await api(`/submissions?limit=${PAGE}&offset=${h.offset}`);
    if (!data.items.length && h.offset === 0) {
      view.replaceChildren(el('h1', { text: 'Lịch sử' }),
        el('div', { class: 'empty', text: 'Chưa có lượt tóm tắt nào. Chạy một bài ở trang So sánh để bắt đầu.' }));
      return;
    }
    const pair = (sub, field, fmt) => ORDER.map((key) => {
      const r = sub.results.find((x) => x.model_key === key);
      return r && !r.error && r[field] != null ? fmt(r[field]) : '–';
    }).join(' / ');
    const voteText = (v) => (v ? ({ seq2seq: 'Seq2Seq', vit5: 'ViT5', tie: 'Ngang nhau' })[v.choice] + (v.blind ? ' (mù)' : '') : '–');
    const body = data.items.map((sub) => el('tr', {},
      el('td', { text: when(sub.created_at) }),
      el('td', {}, el('button', { type: 'button', class: 'linklike', text: sub.title || `Bài ${sub.id} (${sub.n_words} từ)`,
        onclick: () => openSubmission(sub.id) })),
      el('td', { text: sub.source === 'testset' ? 'Bài mẫu' : 'Tự nhập' }),
      el('td', { text: pair(sub, 'latency_ms', (v) => nf(v / 1000, 2)) }),
      el('td', { text: pair(sub, 'rougeL', (v) => nf(v, 1)) }),
      el('td', { text: voteText(sub.vote) })));
    const end = Math.min(h.offset + PAGE, data.total);
    view.replaceChildren(el('h1', { text: 'Lịch sử' }),
      el('div', { class: 'table-wrap' }, el('table', {},
        el('thead', {}, el('tr', {}, ['Thời gian', 'Bài báo', 'Nguồn', 'Giây (Seq2Seq / ViT5)', 'ROUGE-L (Seq2Seq / ViT5)', 'Bình chọn']
          .map((t) => el('th', { text: t })))),
        el('tbody', {}, body))),
      el('div', { class: 'pager' },
        el('button', { type: 'button', class: 'ghost', text: 'Trước', disabled: h.offset === 0,
          onclick: () => { h.offset = Math.max(0, h.offset - PAGE); viewHistory(); } }),
        el('span', { text: `${h.offset + 1}–${end} trên ${data.total}` }),
        el('button', { type: 'button', class: 'ghost', text: 'Sau', disabled: end >= data.total,
          onclick: () => { h.offset += PAGE; viewHistory(); } })));
  }

  async function openSubmission(id) {
    try {
      const sub = await api('/submissions/' + id);
      state.form = { title: sub.title, article: sub.article, reference: sub.reference || '', sample_id: sub.sample_id };
      state.current = sub;
      goto('compare');
    } catch (e) { alert('Không mở được lượt này: ' + e.message); }
  }

  // ---------- trang Thống kê ----------
  async function viewStats() {
    const data = await api('/stats');
    const rows = [
      ['Số lần chạy', (m) => m.runs],
      ['Thời gian trung bình', (m) => secs(m.avg_latency_ms)],
      ['Đầu vào bị cắt', (m) => (m.truncated_rate == null ? '–' : nf(m.truncated_rate * 100, 1) + '%')],
      ['Số lượt có tham chiếu', (m) => m.scored],
      ['ROUGE-1 trung bình', (m) => nf(m.avg_rouge1)],
      ['ROUGE-2 trung bình', (m) => nf(m.avg_rouge2)],
      ['ROUGE-L trung bình', (m) => nf(m.avg_rougeL)],
      ['Lần lỗi', (m) => m.errors],
    ];
    const table = el('div', { class: 'table-wrap' }, el('table', {},
      el('thead', {}, el('tr', {}, el('th', { text: 'Chỉ số' }),
        ...ORDER.map((k) => el('th', { class: k === 'seq2seq' ? 'seq' : 'vit', text: MODELS[k].label })))),
      el('tbody', {}, rows.map(([name, fn]) => el('tr', {}, el('td', { text: name }),
        ...ORDER.map((k) => el('td', { text: String(fn(data.models[k])) })))))));

    const v = data.votes;
    const seg = (cls, label, n) => (n ? el('div', { class: cls, style: `flex:${n}`, text: `${label} ${n}` }) : null);
    const votes = v.total
      ? el('div', {}, el('div', { class: 'bar' }, seg('b-seq', 'Seq2Seq', v.seq2seq), seg('b-tie', 'Ngang nhau', v.tie), seg('b-vit', 'ViT5', v.vit5)),
        el('p', { class: 'muted', text: `${v.total} bình chọn, trong đó ${v.blind} ở chế độ mù.` }))
      : el('div', { class: 'empty', text: 'Chưa có bình chọn nào. Chọn bản tốt hơn sau mỗi lượt tóm tắt để xem tổng hợp ở đây.' });

    view.replaceChildren(el('h1', { text: 'Thống kê' }),
      el('p', { class: 'muted', text: `${data.submissions} lượt tóm tắt đã lưu. ROUGE chỉ tính trên các lượt có bản tham chiếu.` }),
      table, el('h2', { class: 'section', text: 'Bình chọn của người đánh giá' }), votes);
  }

  // ---------- điều hướng ----------
  const ROUTES = { compare: viewCompare, samples: viewSamples, history: viewHistory, stats: viewStats };
  function goto(name) {
    if (location.hash === '#' + name) route(); else location.hash = '#' + name;
  }
  function route() {
    const name = ROUTES[location.hash.slice(1)] ? location.hash.slice(1) : 'compare';
    document.querySelectorAll('#nav a').forEach((a) =>
      a.setAttribute('aria-current', a.getAttribute('href') === '#' + name ? 'page' : 'false'));
    Promise.resolve(ROUTES[name]()).catch((e) =>
      view.replaceChildren(el('p', { class: 'error', text: 'Không tải được trang: ' + e.message })));
  }
  window.addEventListener('hashchange', route);
  refreshStatus();
  route();
})();
