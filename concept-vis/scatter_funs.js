/**
 * Functions for drawing the scatterplot
 */


function update_images(im_ids, name, plural, offset) {
  currentWidth = parseInt(d3.select('#div_main').style('width'), 10)
  var initial = ratio*scales.x(Math.max.apply(null, values0)) + ((currentWidth  -  (ratio *currentWidth))/3 - 115)

  var sel = eval(name + plural).selectAll("." + name)
    .data(im_ids, (d) => d)
    .enter()
    .append("svg:image")
    .attrs({
      "x": initial + offset + 30,
      "height": 100,
      "width": 100,
      "xlink:href": (d) => d

    })
    .classed(name, true);

    sel.attr("y", function(d, i) { return (100 + 10) * i + 100});    
    return sel
}



