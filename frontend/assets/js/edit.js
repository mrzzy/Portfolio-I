/*
 * assets/js/edit.js
 * Pasitche Frontend
 * Client side JS for Edit Page
*/


/* Read the data from the given file selected by the given file input element
 * Reads the data in the file data type given by type
 * Calls the given onread callback when the file is sucessfully read
 * NOTE: assumes only one file is selected with input
*/
FileDataType = { binary: 0, url: 1 };
function readFileInput(input, type, onread) {
    if(input.files.length != 1) throw "No or multiple files selected";

    // Setup file reader
    const file =  input.files[0];
    reader = new FileReader();
    reader.onload = (event) => {
        // Call callback with file data
        onread(event.target.result);
    }

    // Read file for given file data type
    if(type == FileDataType.binary) {
        reader.readAsBinaryString(file);
    } else if (type == FileDataType.url) {
        reader.readAsDataURL(file);
    } else {
        throw "Unknown file data type given";
    }
}

/* Render the image pickers on this page with the selected images */
function renderImagePickers() {
    const styleFileInput = $(".style.picker input[type='file']").get(0);
    const contentFileInput = $(".content.picker input[type='file']").get(0);
    
    const renderImage = (selector) => (
        (imageData) => {
            const imageElement = $(selector).find("img");
            imageElement.attr("src", imageData);
        }
    );

    if(styleFileInput.files.length == 1) {
        readFileInput(styleFileInput, FileDataType.url, renderImage(".picker.style"));
    } 
    if(contentFileInput.files.length == 1) {
        readFileInput(contentFileInput, FileDataType.url, renderImage(".picker.content"));
    }
}

$(document).ready(() => {
    renderImagePickers();

    // Show file picker on click
    $(".picker input[type='file']").click(function (event) {
        event.stopPropagation();
    });
    $(".picker").click(function (event) {
        $(this).children("input[type='file']").click();
    });

    // Update page with selected image on user selection of image
    $(".picker").change(function (event) {
        renderImagePickers();
    });

    // Trigger style transfer on server
    $("button#style-transfer").click(function (event) {
        const styleFileInput = $(".style.picker input[type='file']").get(0);
        const contentFileInput = $(".content.picker input[type='file']").get(0);

        // Check if use has select the required files
        if(styleFileInput.files.length != 1 || contentFileInput.files.length != 1) {
            alert("Please select content & style images");
            return;
        }
    
        readFileInput(styleFileInput, FileDataType.binary, (styleImageData) => {
            readFileInput(contentFileInput, FileDataType.binary, (contentImageData) => {
                // Send style transfer request
                const requestJSON = buildTransferRequest(contentImageData,styleImageData);
                fetch("http://localhost:8989/api/style", {
                    method: "POST",
                    body: requestJSON
                })
                .then((response) => {
                    // Check response status
                    if(response.status != 200) {
                        throw "TransferRequestError: Could not submit style transfer request";
                    }

                    return response.json();
                })
                .then((transferResponse) => {
                    // Wait for style transfer to complete on server
                    const taskID = transferResponse.id
                    console.log("Registered task:" + taskID);

                    window.setInterval(() => {
                        checkTransferProgress(taskID, (progress) => console.log(progress));
                    }, 1000);
                });
            });
        });
    });
});

// loading bar
$('.count').each(function () {
    $(this).prop('Counter',0).animate({
        Counter: $(this).text()
    }, {
        duration: 4000,
        easing: 'swing',
        step: function (now) {
            $(this).text(Math.ceil(now));
        }
    });
});
