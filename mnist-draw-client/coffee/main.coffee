fabric = require("fabric").fabric
$ = require("jquery")
window.jQuery = $
bs = require("bootstrap")

resultStrings = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

$(document).ready () ->
    canvas = new fabric.Canvas("paper", isDrawingMode: true)
    canvas.freeDrawingBrush.width = 15

    $("#result-container").hide()

    $("#clear-button").click () ->
        canvas.clear()
        $("#result-container").hide()

    $("#open-button").click () ->
        if not window.localStorage?
            alert("Unfortunately, your browser is not supported")
            return
        url = canvas.toDataURLWithMultiplier("png", 28 / 200)
        window.open(url)

    $("#send-button").click () ->
        if not window.localStorage?
            alert("Unfortunately, your browser is not supported")
            return
        url = canvas.toDataURLWithMultiplier("png", 28 / 200)

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
            mnist_data = []
            i = 0
            while i < img_data.length
                alpha = img_data[i + 3]
                mnist_data.push(alpha / 255)
                i += 4
            return mnist_data
        .then (mnist) ->
            data = JSON.stringify(mnist)
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
            $("#result-container").show()
            if result?
                $("#result").text(resultStrings[result])
            else
                $("#result").text("Not a number. This AI isn't the brightest, fyi. Your drawing may also suck.")
        .catch (error) ->
            alert(error)