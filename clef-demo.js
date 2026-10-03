// Clef 自動操作デモ UI。
//
// このモジュールはゲーム物理コード（index.html）を一切変更しない。index.html
// から渡される {observe, capture, beginAuto, execute, endAuto} だけを使って、
// 観測 → 画像 → POST /api/decide → 選択実行、のループを回す。DOM 構築・fetch・
// 状態表示はすべてここに閉じる。

const STEP_TICKS = { fine: 12, medium: 30, coarse: 72 };

export function mountClefDemo({ observe, capture, beginAuto, execute, endAuto }) {
  const style = document.createElement('style');
  style.textContent = `
    #clef-panel {
      position: fixed; right: 16px; top: 16px; z-index: 10;
      width: 300px; max-height: 90vh; overflow-y: auto;
      padding: 12px 14px; border-radius: 12px;
      background: rgba(0,0,0,.6); color: #fff; line-height: 1.5;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px; backdrop-filter: blur(8px); user-select: none;
    }
    #clef-panel b { font-size: 15px; }
    #clef-panel .clef-row { margin-top: 6px; }
    #clef-panel button {
      font-size: 13px; padding: 6px 10px; margin-right: 6px; margin-top: 8px;
      border-radius: 8px; border: 1px solid rgba(255,255,255,.35);
      background: rgba(255,255,255,.08); color: #fff; cursor: pointer;
    }
    #clef-panel button:disabled { opacity: .4; cursor: default; }
    #clef-panel button:not(:disabled):hover { background: rgba(255,255,255,.18); }
    #clef-status { color: #9ee7ff; }
    #clef-model { opacity: .8; font-size: 12px; }
    #clef-probabilities { font-size: 12px; margin-top: 4px; }
    #clef-probabilities .clef-bar-row { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
    #clef-probabilities .clef-bar-label { width: 64px; flex: none; opacity: .85; }
    #clef-probabilities .clef-bar-track { flex: 1; height: 8px; background: rgba(255,255,255,.12); border-radius: 4px; overflow: hidden; }
    #clef-probabilities .clef-bar-fill { height: 100%; background: #6ee7a8; }
    #clef-probabilities .clef-bar-fill.chosen { background: #ffc857; }
    #clef-probabilities .clef-bar-value { width: 40px; flex: none; text-align: right; opacity: .85; }
    #clef-history { font-size: 12px; margin-top: 4px; opacity: .9; }
    #clef-history div { margin-top: 2px; border-top: 1px solid rgba(255,255,255,.12); padding-top: 2px; }
  `;
  document.head.appendChild(style);

  const panel = document.createElement('div');
  panel.id = 'clef-panel';
  panel.innerHTML = `
    <b>Clef 自動操作デモ</b><br>
    <span id="clef-model" class="clef-row">モデル確認中...</span>
    <div class="clef-row">
      <button id="clef-start" disabled>開始</button>
      <button id="clef-stop" disabled>停止</button>
      <button id="clef-reset">リセット</button>
    </div>
    <div class="clef-row">状態: <span id="clef-status">初期化中</span></div>
    <div class="clef-row">判断回数: <span id="clef-count">0</span></div>
    <div class="clef-row">クレーン位置: <span id="clef-position">-</span></div>
    <div class="clef-row">把持ID: <span id="clef-held">-</span></div>
    <div class="clef-row">直近操作: <span id="clef-last-action">-</span></div>
    <div id="clef-probabilities" class="clef-row"></div>
    <div class="clef-row">履歴 (直近5件):</div>
    <div id="clef-history"></div>
  `;
  document.body.appendChild(panel);

  const els = {
    model: panel.querySelector('#clef-model'),
    start: panel.querySelector('#clef-start'),
    stop: panel.querySelector('#clef-stop'),
    reset: panel.querySelector('#clef-reset'),
    status: panel.querySelector('#clef-status'),
    count: panel.querySelector('#clef-count'),
    position: panel.querySelector('#clef-position'),
    held: panel.querySelector('#clef-held'),
    lastAction: panel.querySelector('#clef-last-action'),
    probabilities: panel.querySelector('#clef-probabilities'),
    history: panel.querySelector('#clef-history'),
  };

  let healthReady = false;
  let running = false;
  let resetting = false;
  let generation = 0;
  let decisionCount = 0;
  let historyLog = [];
  let inFlightFetch = null;

  function setStatus(text) {
    els.status.textContent = text;
  }

  function updateButtons() {
    els.start.disabled = running || resetting || !healthReady || inFlightFetch !== null;
    els.stop.disabled = !running || resetting;
    els.reset.disabled = resetting;
  }

  function renderPosition(obs) {
    els.position.textContent = `x=${obs.claw.position.x.toFixed(2)}, z=${obs.claw.position.z.toFixed(2)}`;
    els.held.textContent = obs.claw.held_prize_id === null ? '-' : String(obs.claw.held_prize_id);
  }

  function renderDecision(actionAns, stepAns, elapsedMs) {
    els.lastAction.textContent =
      `action=${actionAns.choice} (${(actionAns.confidence * 100).toFixed(1)}%) / ` +
      `step=${stepAns.choice} (${(stepAns.confidence * 100).toFixed(1)}%) / ${elapsedMs.toFixed(0)}ms`;

    const bars = (title, ans) => {
      const rows = Object.entries(ans.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([name, p]) => `
          <div class="clef-bar-row">
            <span class="clef-bar-label">${name}</span>
            <span class="clef-bar-track"><span class="clef-bar-fill${name === ans.choice ? ' chosen' : ''}" style="width:${Math.max(0, Math.min(1, p)) * 100}%"></span></span>
            <span class="clef-bar-value">${(p * 100).toFixed(1)}%</span>
          </div>`)
        .join('');
      return `<div><b>${title}</b></div>${rows}`;
    };
    els.probabilities.innerHTML = bars('action', actionAns) + bars('step', stepAns);
  }

  function renderHistory() {
    els.history.innerHTML = historyLog
      .map((h) => `<div>${h.action}/${h.step}: (${h.before.phase} ${h.before.claw.x.toFixed(1)},${h.before.claw.z.toFixed(1)} s${h.before.score}) → (${h.after.phase} ${h.after.claw.x.toFixed(1)},${h.after.claw.z.toFixed(1)} s${h.after.score})</div>`)
      .join('');
  }

  function updateCount() {
    els.count.textContent = String(decisionCount);
  }

  function fail(myGen, message) {
    if (myGen !== generation) return;
    setStatus(`エラー: ${message}`);
    endAuto();
    running = false;
    updateButtons();
  }

  async function runLoop(myGen) {
    try {
      await beginAuto();
    } catch (err) {
      if (myGen !== generation) return;
      fail(myGen, `開始待機失敗: ${err.message}`);
      return;
    }
    if (myGen !== generation) return;

    while (myGen === generation) {
      const before = observe();

      if (before.prizes.every((p) => p.scored)) {
        setStatus('完了（全景品回収）');
        endAuto();
        running = false;
        updateButtons();
        return;
      }

      renderPosition(before);
      setStatus('推論中...');

      const image = capture();
      // モデルへは最小限の情報のみ送る: phase, has_prize(=held_prize_id有無), image, 簡略履歴。
      // 座標・速度・得点・景品リストなどの詳細観測はローカル(HUD/historyLog)にのみ保持する。
      const requestBody = {
        phase: before.phase,
        has_prize: before.claw.held_prize_id !== null,
        image,
        history: historyLog.slice().map(({ action, step }) => ({ action, step })),
      };

      let fetchPromise;
      try {
        fetchPromise = fetch('/api/decide', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
      } catch (err) {
        fail(myGen, `リクエスト失敗: ${err.message}`);
        return;
      }
      inFlightFetch = fetchPromise;
      fetchPromise.finally(() => {
        if (inFlightFetch === fetchPromise) inFlightFetch = null;
        updateButtons();
      });

      let res;
      try {
        res = await fetchPromise;
      } catch (err) {
        if (myGen !== generation) return;
        fail(myGen, `通信エラー: ${err.message}`);
        return;
      }
      if (myGen !== generation) return;

      if (!res.ok) {
        let detail = '';
        try { detail = await res.text(); } catch { /* ignore */ }
        fail(myGen, `HTTP ${res.status} ${detail}`.trim());
        return;
      }

      let data;
      try {
        data = await res.json();
      } catch (err) {
        if (myGen !== generation) return;
        fail(myGen, `応答の解析に失敗: ${err.message}`);
        return;
      }
      if (myGen !== generation) return;

      const actionAns = data?.answers?.action;
      const stepAns = data?.answers?.step;
      if (!actionAns || typeof actionAns.choice !== 'string' || !stepAns || typeof stepAns.choice !== 'string') {
        fail(myGen, '不正な応答形式 (answers.action/step が見つからない)');
        return;
      }
      const ticks = STEP_TICKS[stepAns.choice];
      if (ticks === undefined) {
        fail(myGen, `不正な step choice: ${stepAns.choice}`);
        return;
      }

      decisionCount += 1;
      updateCount();
      renderDecision(actionAns, stepAns, data.elapsed_ms ?? 0);
      setStatus('操作中...');

      try {
        await execute(actionAns.choice, { ticks });
      } catch (err) {
        if (myGen !== generation) return; // cancelled by stop/reset: expected rejection
        fail(myGen, `操作失敗: ${err.message}`);
        return;
      }
      if (myGen !== generation) return;

      const after = observe();
      historyLog.push({
        action: actionAns.choice,
        step: stepAns.choice,
        before: { phase: before.phase, claw: { x: before.claw.position.x, z: before.claw.position.z }, score: before.score },
        after: { phase: after.phase, claw: { x: after.claw.position.x, z: after.claw.position.z }, score: after.score },
      });
      if (historyLog.length > 5) historyLog.shift();
      renderHistory();
      renderPosition(after);
    }
  }

  function start() {
    if (running || resetting || !healthReady || inFlightFetch !== null) return;
    running = true;
    generation += 1;
    const myGen = generation;
    historyLog = [];
    decisionCount = 0;
    updateCount();
    renderHistory();
    updateButtons();
    runLoop(myGen);
  }

  function stop() {
    if (!running) return;
    generation += 1;
    endAuto();
    running = false;
    setStatus(inFlightFetch ? '停止（応答待ち）' : '停止');
    updateButtons();
  }

  async function reset() {
    if (resetting) return;
    resetting = true;
    generation += 1;
    endAuto();
    running = false;
    setStatus('リセット待ち...');
    updateButtons();

    const pending = inFlightFetch;
    if (pending) {
      try { await pending; } catch { /* discard stale response */ }
    }
    location.reload();
  }

  els.start.addEventListener('click', start);
  els.stop.addEventListener('click', stop);
  els.reset.addEventListener('click', reset);

  (async () => {
    setStatus('モデル確認中...');
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data.ready) throw new Error('モデル未準備');
      healthReady = true;
      els.model.textContent = `${data.model} (${String(data.revision).slice(0, 8)}, ${data.device})`;
      setStatus('待機中');
    } catch (err) {
      healthReady = false;
      els.model.textContent = 'モデル利用不可';
      setStatus(`モデル起動エラー: ${err.message} — 'uv run --no-sync python main.py' でサーバーを起動してください`);
    }
    renderPosition(observe());
    updateButtons();
  })();
}
