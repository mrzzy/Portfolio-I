/*
 * assets/js/edit.js
 * Pasitche Frontend
 * Client side JS for Edit Page
*/


/* Read the data from the given file selected by the given file input element
 * Calls the given onread callback when the file is sucessfully read
 * NOTE: assumes only one file is selected with input
*/
function readFileInput(input, onread) {
    if(input.files.length != 1) throw "No or multiple files selected";

    // Read file with HTML5 file reader
    const file =  input.files[0];
    reader = new FileReader();
    reader.onload = (event) => {
        // Call callback with file data
        onread(event.target.result);
    }
    reader.readAsBinaryString(file);
}

$(document).ready(() => {
    $(".inputfile").on("change", (event) => {
        readFileInput(event.target, (data) => console.log("got data: ", data));
    });
});
