/*
 * server.js
 * Pastiche
 * Front End Server
*/

const express = require('express');

const app = express()
const server_port = 8080

// Static Assets
app.use('/assets', express.static(__dirname + '/assets'));

// Frontend Routes
/* Homepage/Landing route, display homepage index.html */
app.get('/', (request, response) => {
    response.sendFile(__dirname + "/public/index.html");
});

/* Edit route, display edit.html */
app.get('/edit', (request, response) => {
    response.sendFile(__dirname + "/public/edit.html");
});

/* Error Handling Middleware
 * Handles exceptions thrown in route handlers
 * Logs the exception
 * Returns a 500 status to the client
*/
app.use((err, request, response, next) => {
      console.error("FATAL: Internal error", err)
      response.status(500).send('Something broke!')
});


// Start server
app.listen(server_port, (err) => {
    if (err) { return console.error('FATAL: server failed to start', err); }

    console.log(`pastiche front-end server is listening on ${server_port}`);
});
