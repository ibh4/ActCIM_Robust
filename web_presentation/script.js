/* ================= 幻灯片翻页控制（录屏友好） ================= */
(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const toc = document.getElementById("toc");
  const pageNow = document.getElementById("pageNow");
  const pageTotal = document.getElementById("pageTotal");
  const progressFill = document.getElementById("progressFill");
  let idx = 0;

  pageTotal.textContent = slides.length;

  // 顶部目录按钮
  slides.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.textContent = s.dataset.title || `第${i + 1}页`;
    btn.addEventListener("click", () => go(i));
    toc.appendChild(btn);
  });
  const tocBtns = Array.from(toc.children);

  function go(i) {
    idx = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((s, k) => s.classList.toggle("active", k === idx));
    tocBtns.forEach((b, k) => b.classList.toggle("active", k === idx));
    tocBtns[idx].scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
    pageNow.textContent = idx + 1;
    progressFill.style.width = ((idx + 1) / slides.length) * 100 + "%";
    // 支持 #page-N 直达，便于录屏分段
    history.replaceState(null, "", "#page-" + (idx + 1));
  }

  // 键盘：→ / 空格 / PgDn 下一页；← / PgUp 上一页；Home/End；F 全屏
  document.addEventListener("keydown", (e) => {
    if (["ArrowRight", " ", "PageDown"].includes(e.key)) { e.preventDefault(); go(idx + 1); }
    else if (["ArrowLeft", "PageUp"].includes(e.key)) { e.preventDefault(); go(idx - 1); }
    else if (e.key === "Home") { e.preventDefault(); go(0); }
    else if (e.key === "End") { e.preventDefault(); go(slides.length - 1); }
    else if (e.key.toLowerCase() === "f") {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    }
  });

  // 点击页面右/左 1/4 区域翻页（不影响导航栏与图片放大查看）
  document.getElementById("deck").addEventListener("click", (e) => {
    if (e.target.closest("a, button, img, pre, table")) return;
    const x = e.clientX / window.innerWidth;
    if (x > 0.75) go(idx + 1);
    else if (x < 0.25) go(idx - 1);
  });

  // 点击图片放大 / 再点还原（录屏时展示细节）
  document.querySelectorAll("img.fig").forEach((img) => {
    img.addEventListener("click", () => {
      let overlay = document.getElementById("figOverlay");
      if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "figOverlay";
        Object.assign(overlay.style, {
          position: "fixed", inset: "0", background: "rgba(10,20,36,0.92)",
          display: "flex", alignItems: "center", justifyContent: "center",
          zIndex: "100", cursor: "zoom-out",
        });
        overlay.addEventListener("click", () => overlay.remove());
        document.body.appendChild(overlay);
      }
      overlay.replaceChildren();
      const big = document.createElement("img");
      big.src = img.src;
      Object.assign(big.style, {
        maxWidth: "94vw", maxHeight: "92vh", objectFit: "contain",
        borderRadius: "8px", boxShadow: "0 10px 50px rgba(0,0,0,0.5)",
        background: "#fff",
      });
      overlay.appendChild(big);
    });
    img.style.cursor = "zoom-in";
  });

  // 支持 URL hash 直达
  const m = location.hash.match(/^#page-(\d+)$/);
  go(m ? parseInt(m[1], 10) - 1 : 0);
})();
