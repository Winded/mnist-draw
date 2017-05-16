fabric = require("fabric").fabric
$ = require("jquery")
window.jQuery = $
bs = require("bootstrap")

resultStrings = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
randomStrings = ["a bird", "a plane", "superman", "your bad drawing"]

ratingEnabled = false
lastMnistData = []

imageToMnistData = (img_data) ->
    mnist_data = []
    i = 0
    while i < img_data.length
        alpha = img_data[i + 3]
        mnist_data.push(alpha / 255)
        i += 4
    return mnist_data

sendData = (data) ->
    new Promise (success, failure) ->
        xhr = new XMLHttpRequest()
        xhr.open("POST", "http://192.168.0.13:8000/", true)
        xhr.setRequestHeader("Content-type", "application/json")
        xhr.onreadystatechange = () ->
            if xhr.readyState == 4
                json = JSON.parse(xhr.responseText)
                if xhr.status == 200
                    success(json.result)
                else
                    failure(json.message)
        xhr.send(data)

$(document).ready () ->
    canvasCtx = $("#paper")[0].getContext("2d")
    canvasCtx.canvas.width = Math.min(document.documentElement.clientWidth, 500);
    canvasCtx.canvas.height = canvasCtx.canvas.width;

    canvas = new fabric.Canvas("paper", isDrawingMode: true)
    canvas.freeDrawingBrush.width = Math.ceil(canvasCtx.canvas.width / 20)

    $("#draw-container").show()
    $("#loading-container").hide()
    $("#result-container").hide()

    $("#clear-button").click () ->
        canvas.clear()
        $("#result-response").text("")

    $("#open-button").click () ->
        if not window.localStorage?
            alert("Unfortunately, your browser is not supported")
            return
        url = canvas.toDataURLWithMultiplier("png", 28 / 200)
        window.open(url)

    $("#yes-button").click () ->
        $("#draw-container").show()
        $("#loading-container").hide()
        $("#result-container").hide()
        $("#footer").css("height", "60px")
        if ratingEnabled
            $("#result-response").text("I knew it!")
            sendData(JSON.stringify({data: lastMnistData, correct: true}))
        else
            $("#result-response").text("")

    $("#no-button").click () ->
        $("#draw-container").show()
        $("#loading-container").hide()
        $("#result-container").hide()
        $("#footer").css("height", "60px")
        if ratingEnabled
            $("#result-response").text("Ok :(")
            sendData(JSON.stringify({data: lastMnistData, correct: false}))
        else
            $("#result-response").text("")

    $("#send-button").click () ->
        if not window.localStorage?
            alert("Unfortunately, your browser is not supported")
            return
        url = canvas.toDataURLWithMultiplier("png", 28 / 200)

        $("#draw-container").hide()
        $("#loading-container").show()
        $("#result-container").hide()
        $("#footer").css("height", "60px")

        new Promise (success, failure) ->
            c = document.createElement("canvas")
            c.width = 28
            c.height = 28
            ctx = c.getContext("2d")
            img = new Image()
            img.onload = () ->
                ctx.drawImage(img, 0, 0)
                success(canvas: c, ctx: ctx)
            img.src = url
        .then (c) ->
            img_data = c.ctx.getImageData(0, 0, 28, 28).data
            if not img_data?
                failure("Failed to load image data")
            return imageToMnistData(img_data)
        .then (mnist) ->
            data = JSON.stringify(mnist)
            console.log(data)
            lastMnistData = data
            new Promise (success, failure) ->
                xhr = new XMLHttpRequest()
                xhr.open("POST", "http://192.168.0.13:8000/", true)
                xhr.setRequestHeader("Content-type", "application/json")
                xhr.onreadystatechange = () ->
                    if xhr.readyState == 4
                        json = JSON.parse(xhr.responseText)
                        if xhr.status == 200
                            success(json.result)
                        else
                            failure(json.message)
                xhr.send(data)
        .then (result) ->
            $("#draw-container").hide()
            $("#loading-container").hide()
            $("#result-container").show()
            $("#footer").css("height", "120px")
            if result?
                $("#result").text(resultStrings[result])
                ratingEnabled = true
            else
                $("#result").text(randomStrings[Math.floor(Math.random() * randomStrings.length)])
                ratingEnabled = false
        .catch (error) ->
            $("#draw-container").show()
            $("#loading-container").hide()
            $("#result-container").hide()
            $("#footer").css("height", "60px")
            alert("An error occurred: #{error}")