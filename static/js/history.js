window.addEventListener("load", function () {
  var myModal = new bootstrap.Modal(document.getElementById("popupmodal"), {
    backdrop: "static",
    keyboard: false,
  });
  myModal.show();

  // Show blurred background when modal is shown
  $("#popupmodal").on("show.bs.modal", function () {
    $("#blurBackground").fadeIn();
  });

  // Hide blurred background when modal is hidden
  $("#popupmodal").on("hidden.bs.modal", function () {
    $("#blurBackground").fadeOut();
  });

  document.getElementById("loginForm").addEventListener("submit", function (e) {
    e.preventDefault();
    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;

    if (username === "Admin" && password === "passAdmin90") {
      myModal.hide();
      $("#blurBackground").fadeOut(); // Ensure blur background is removed after login
      $("#mainContent").fadeIn(); // Show the main content after successful login
    } else {
      alert("Invalid credentials. Please try again.");
    }
  });
});
