document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        alert("Please select a file!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("report").textContent = result.report || "Error generating report.";
});
