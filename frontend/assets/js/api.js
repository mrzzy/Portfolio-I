/*
 * assets/js/api.js
 * Pasitche Frontend
 * Rest API JS
*/

const serverURL = "http://localhost:8989";

/* Build a style transfer request for the given content and style image data
 * and optional settings overrides
 * Returns the request serialised in the json format
*/
function buildTransferRequest(contentData, styleData, settings={}) {
    // Base 64 encode image data for transmission
    // TODO: remove hardcoded settings
    settings.n_epochs = 100;
    return JSON.stringify({
        content_image: window.btoa(contentData),
        style_image: window.btoa(styleData),
        settings: settings
    });
}

/* Check the progress the task for the given task ID
 * Passes the progress as a float (0-1.0) to the given callback
*/
function checkTransferProgress(taskID, onCheck) {
    fetch(serverURL + "/api/status/" + taskID)
    .then((response) => {
        // Check that style transfer is running as expected on the server
        if(response.status == 404) {
            throw "Server has disowned style transfer";
        } else if (response.status == 500){
            throw "Server encountered internal error performing content transfer";
        }    
        
        return response.json()
    })
    .then((statusResponse) => onCheck(statusResponse.progress));
}
