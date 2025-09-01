const BACKEND_BASE_URL = "http://localhost:8000";

export async function modifyWindowSize(file, windowId, newWidth, newHeight) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append(
    "modification",
    JSON.stringify({
      window_id: windowId,
      new_width: newWidth,
      new_height: newHeight,
    })
  );

  const response = await fetch(`${BACKEND_BASE_URL}/api/modify-window`, {
    method: "POST",
    body: formData,
  });

  const blob = await response.blob();
  return new File([blob], file.name, { type: file.type });
}