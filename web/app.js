const input = document.getElementById("videoInput");
const player = document.getElementById("videoPlayer");
const meta = document.getElementById("videoMeta");
const clearBtn = document.getElementById("clearBtn");

let currentUrl = null;

function clearVideo() {
  if (currentUrl) {
    URL.revokeObjectURL(currentUrl);
    currentUrl = null;
  }
  player.removeAttribute("src");
  player.load();
  meta.textContent = "No video loaded.";
}

input.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    clearVideo();
    return;
  }

  clearVideo();
  currentUrl = URL.createObjectURL(file);
  player.src = currentUrl;
  meta.textContent = `${file.name} • ${(file.size / 1024 / 1024).toFixed(1)} MB`;
});

clearBtn.addEventListener("click", () => {
  input.value = "";
  clearVideo();
});
