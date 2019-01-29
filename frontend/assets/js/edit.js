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

/* Build a style transfer request for the given content and style image data
 * and optional settings overrides
 * Returns the request serialised in the json format
*/
function buildTransferRequest(contentData, styleData, settings={}) {
    // Base 64 encode image data for transmission
    return JSON.stringify({
        content_image: window.btoa(contentData),
        style_image: window.btoa(styleData),
        settings: settings
    });
}

$(document).ready(() => {
    // Show file picker on click
    $(".picker input[type='file']").click(function (event) {
        event.stopPropagation();
    });
    $(".picker").click(function (event) {
        $(this).children("input[type='file']").click();
    });

    // Update page with selected image on user selection of image
    $(".picker").change(function (event) {
        readFileInput(event.target, FileDataType.url, (imageData) => {
            const imageElement = $(this).find("img");
            imageElement.attr("src", imageData);
        }, FileDataType.url);
    });
});
