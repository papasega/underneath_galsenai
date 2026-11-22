const log = document.getElementById("log"), form = document.getElementById("form");
const input = document.getElementById("input"), send = document.getElementById("send");
const picker = document.getElementById("model"), detail = document.getElementById("detail");
let history = [], busy = false, models = [];

fetch("/api/config").then(r => r.json()).then(c => {
  models = c.models;
  models.forEach(m => picker.add(new Option(m.name + (m.available ? "" : " (indisponible)"), m.id)));
  picker.value = c.default; showDetail();
});
function showDetail() { const m = models.find(x => x.id === picker.value); detail.textContent = m ? m.detail : ""; }
picker.onchange = () => { showDetail(); reset(); };

// ---- Markdown (safe: DOM nodes only, never innerHTML) ----
function inline(parent, text) {
  const re = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\)|\*(?!\s)[^*\n]+?(?<!\s)\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.append(text.slice(last, m.index));
    const t = m[0]; let el;
    if (t.startsWith("**")) { el = document.createElement("strong"); inline(el, t.slice(2, -2)); }
    else if (t.startsWith("`")) { el = document.createElement("code"); el.textContent = t.slice(1, -1); }
    else if (t.startsWith("[")) { const i = t.indexOf("]("); el = document.createElement("a"); el.href = t.slice(i + 2, -1); el.target = "_blank"; el.rel = "noopener noreferrer"; el.textContent = t.slice(1, i); }
    else { el = document.createElement("em"); inline(el, t.slice(1, -1)); }
    parent.append(el); last = m.index + t.length;
  }
  if (last < text.length) parent.append(text.slice(last));
}
const LIST = /^(\s*)([-*•]|\d+[.)])\s+(.*)$/;
function mk(tag, text) { const e = document.createElement(tag); if (text !== undefined) inline(e, text); return e; }
function markdown(el, src) {
  el.replaceChildren();
  const lines = src.replace(/\r/g, "").split("\n");
  let i = 0, para = [], stack = [];
  const flush = () => { if (para.length) { const p = document.createElement("p"); para.forEach((l, k) => { if (k) p.append(document.createElement("br")); inline(p, l); }); el.append(p); para = []; } };
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {                       // fenced code (an unclosed fence while streaming is fine)
      flush(); stack = []; const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      const pre = document.createElement("pre"), code = document.createElement("code"); code.textContent = buf.join("\n"); pre.append(code); el.append(pre); i++; continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flush(); stack = []; el.append(mk("h" + Math.min(h[1].length + 1, 4), h[2])); i++; continue; }
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); stack = []; el.append(document.createElement("hr")); i++; continue; }
    if (/^>\s?/.test(line)) { flush(); stack = []; const q = []; while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, "")); const b = document.createElement("blockquote"); markdown2(b, q.join("\n")); el.append(b); continue; }
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {   // table
      flush(); stack = []; const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++].trim().slice(1, -1).split("|").map(c => c.trim()));
      const t = document.createElement("table");
      rows.filter((_, k) => k !== 1).forEach((r, k) => { const tr = document.createElement("tr"); r.forEach(c => tr.append(mk(k ? "td" : "th", c))); t.append(tr); });
      el.append(t); continue;
    }
    const li = line.match(LIST);
    if (li) {
      flush(); const indent = li[1].replace(/\t/g, "    ").length, ordered = /\d/.test(li[2]);
      while (stack.length && stack[stack.length - 1].indent > indent) stack.pop();
      let top = stack[stack.length - 1];
      if (!top || top.indent < indent || top.ordered !== ordered) {
        if (top && top.indent === indent) stack.pop();
        const list = document.createElement(ordered ? "ol" : "ul"); if (ordered && parseInt(li[2]) > 1) list.start = parseInt(li[2]); const parentLi = stack.length ? stack[stack.length - 1].last : null;
        (parentLi || el).append(list); top = { indent, ordered, list, last: null }; stack.push(top);
      }
      const item = mk("li", li[3]); top.list.append(item); top.last = item; i++; continue;
    }
    if (line.trim() === "") { flush(); i++; continue; }  // keep the list open: numbering continues across blank lines
    if (stack.length && /^\s+\S/.test(line)) { const last = stack[stack.length - 1].last; last.append(document.createElement("br")); inline(last, line.trim()); i++; continue; }
    stack = []; para.push(line); i++;
  }
  flush();
}
const markdown2 = markdown;

function addMsg(role, text, isError = false) {
  document.getElementById("empty")?.remove();
  const row = document.createElement("div"); row.className = "msg " + role;
  const b = document.createElement("div"); b.className = "bubble" + (isError ? " error" : "");
  if (role === "assistant") markdown(b, text); else b.textContent = text;
  row.append(b); log.append(row); log.scrollTop = log.scrollHeight;
  return b;
}

async function ask(text) {
  history.push({ role: "user", content: text });
  addMsg("user", text);
  const bubble = addMsg("assistant", ""); bubble.classList.add("typing");
  const messages = history;  // the system prompt (Gemma only) is added by the server
  let answer = "";
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages, model: picker.value }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const reader = r.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n"); buf = lines.pop();
      for (const l of lines) {
        if (!l.startsWith("data: ") || l.includes("[DONE]")) continue;
        const delta = JSON.parse(l.slice(6)).choices?.[0]?.delta || {};
        answer += delta.content || "";
        bubble.classList.toggle("thinking", !answer && !!delta.reasoning);   // Gemma thinks first
        if (!answer && delta.reasoning) bubble.textContent = "Réflexion…"; else markdown(bubble, answer);
        log.scrollTop = log.scrollHeight;
      }
    }
    history.push({ role: "assistant", content: answer });
  } catch (e) {
    history.pop(); bubble.classList.add("error"); bubble.textContent = "Erreur : " + e.message;
  }
  bubble.classList.remove("typing");
}

async function submit(text) {
  text = text.trim(); if (!text || busy) return;
  busy = send.disabled = true; await ask(text); busy = send.disabled = false; input.focus();
}
form.addEventListener("submit", e => { e.preventDefault(); const t = input.value; input.value = ""; input.style.height = "auto"; submit(t); });
input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = input.scrollHeight + "px"; });
document.querySelectorAll("#themes li").forEach(li => li.onclick = () => submit(li.dataset.q));  // the model answers a natural question
function reset() { history = []; location.reload(); }
document.getElementById("clear").onclick = reset;
