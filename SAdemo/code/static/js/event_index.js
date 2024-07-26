let dropZone = document.getElementById("dropZone");
let fileInput = document.getElementById("fileInput");
let send = document.getElementById("send");
let uploadForm = document.getElementById("uploadForm");

// Thiết lập chỉ chấp nhận các file ảnh
fileInput.setAttribute("accept", "image/*");

let submit = false;
dropZone.addEventListener("click", function () {
    if(!submit){
        fileInput.click(); 
    }
});

fileInput.addEventListener("change", function () {
    if (fileInput.files.length === 1 && isImageFile(fileInput.files[0])) {
        // submit = true;
        // uploadForm.submit();
    } else {
        alert("Please upload exactly one image file.");
        fileInput.value = "";
    }
});

dropZone.addEventListener("dragover", function (e) {
  e.preventDefault();
  this.classList.add("dragover");
});

dropZone.addEventListener("dragleave", function (e) {
  this.classList.remove("dragover");
});

dropZone.addEventListener("drop", function (e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.remove("dragover");


  // Xử lý chỉ khi có đúng một file và đó là ảnh
  if (e.dataTransfer.files.length === 1 && isImageFile(e.dataTransfer.files[0])) {
    fileInput.files = e.dataTransfer.files;
    // submit = true;
    // uploadForm.submit();
  } else {
    alert("Please drop exactly one image file.");
  }
});

send.addEventListener("click", function () {
    if(!submit){
        submit = true;
        alert("Please waiting 1 minute.");
        uploadForm.submit();
    }
});

// Hàm kiểm tra file có phải là file ảnh không
function isImageFile(file) {
    return file.type.startsWith("image/");
}