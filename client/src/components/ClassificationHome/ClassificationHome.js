import React, { useEffect, useState } from "react";

const ClassificationHome = () => {
  const [imgs, setImgs] = useState();

  const [isSuccess, setIsSuccess] = useState(false);

  function uploadImage(e) {
    e.preventDefault();
    var imageFormElement = document.getElementById("image-upload").files[0];

    let request = new XMLHttpRequest();
    let formData = new FormData();

    formData.append("image", imageFormElement);

    request.onload = () => {
      var response = request.response;
      if (response) {
        setIsSuccess(true);
        setImgs(<img src={`data:image/jpeg;base64,${response}`} alt="dogs" />);
      }
    };

    request.open("POST", "http://127.0.0.1:9999/api/infer");
    request.send(formData);
  }
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        marginTop: "3rem",
      }}
    >
      <form id="uploadForm" onSubmit={uploadImage}>
        <input id="image-upload" type="file" name="image" />
        <button type="submit">Upload</button>
      </form>
      {isSuccess === true ? (
        <p style={{ color: "green", fontWeight: "bold", marginBottom: "1rem" }}>
          Success
        </p>
      ) : (
        <></>
      )}
      <div>{imgs}</div>
    </div>
  );
};

export default ClassificationHome;
