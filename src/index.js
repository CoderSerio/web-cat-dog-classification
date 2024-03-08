const input = document.getElementsByTagName("input")[0];
const button = document.getElementsByTagName("button")[0];
const image = document.getElementsByTagName("img")[0];
const specie = document.getElementById("specie");

let file = null;

const predict = async () => {
  const formData = new FormData();
  formData.append("image", file);
  try {
    fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        const res = data.prediction[0];
        const text = res[0] > res[1] ? "CAT" : "DOG";
        console.log(res[0], res[1]);
        specie.innerHTML = text;
      });
  } catch (err) {
    throw err;
  }
};

input.addEventListener("change", (e) => {
  file = e.target.files[0];
  if (file) {
    specie.innerHTML = "?";
    const reader = new FileReader();
    reader.onload = (e) => {
      image.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

button.addEventListener("click", () => {
  if (file) {
    predict(file);
  }
});
