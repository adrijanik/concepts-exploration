/**
 * Experiment with Interactive Concepts Discovery
 */
// http://bl.ocks.org/WilliamQLiu/76ae20060e19bf42d774

var Svg = d3.select("#div_main")
  .append("svg")
  .attr("id", "svg")
  .attr("height", 1000)

var min_x = -3724.8286437539355;
var max_x = 5547.271176306559;
var min_y = -2146.332386264602; 
var max_y = 4272.263057522139;


//var min_x = -16;
//var max_x = 16;
//var min_y = -16;
//var max_y = 16;


var scales = {
  "x": d3.scaleLinear()
    .domain([min_x, max_x])
    .range([50, 2000]),
  "y": d3.scaleLinear()
    .domain([min_y, max_y])
    .range([30, 865]),
  "color": d3.scaleSequential(d3.interpolateRdBu)
    .domain([0, 1])
};

var ratio = 0.6;

var groups = ["heatmap", "points", "masks", "preds", "patches", "svg-quant", "table"]
Svg.selectAll("g")
  .data(groups).enter()
  .append("g")
  .attr("id", (d) => d)

var points = Svg.select("#points"),
    heatmap = Svg.select("#heatmap"),
    masks = Svg.select("#masks"),
    preds = Svg.select("#preds"),
    patches = Svg.select("#patches");

/**
 * Add legend to the chart
 */



var legend = Svg.select("#svg-quant");

legend.append("g")
  .attr("class", "legendLinear")
  .attr("transform", "translate(132,900)");

var legendLinear = d3.legendColor()
  .shapeWidth(35)
  .shapeHeight(30)
  .cells([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
  .orient('horizontal')
  .scale(scales.color);

legend.select(".legendLinear")
  .call(legendLinear);

// draw the scatterplot points
var points = points.selectAll("circle")
  .data(points_data)
  .enter()
  .append("circle")
  .attrs({
    "cy": (d) => scales.y(d.coords[1]),
    "r": 3,
    "fill": (d) => scales.color(d.iou),
    "stroke": "black",
    "stroke-width": 1
  });


// draw the background heatmap
var ux = [... new Set(hm_data.map((d) => d.coords[0]))],
    uy = [... new Set(hm_data.map((d) => d.coords[1]))];

var heatmap = heatmap.selectAll("rect")
  .data(hm_data).enter()
  .append("rect")
  .attrs({
    "y": (d) => scales.y(d.coords[1]),
    "height": scales.y(uy[1]) - scales.y(uy[0]),
    "fill": (d) => scales.color(d.iou),
    "opacity": 0.7
  });


var values0 = points_data.map(function(dict){
  return dict['coords'][0];
}); 
var values1 = points_data.map(function(dict){
  return dict['coords'][1];
}); 


// A function that finishes to draw the chart for a specific device size.
function drawChart() {

  var patch = patches.selectAll(".patch");
  patch.remove();
  var mask = masks.selectAll(".mask");
  mask.remove();
  var pred = preds.selectAll(".pred");
  pred.remove();

	
  // get the current width of the div where the chart appear, and attribute it to Svg
  currentWidth = parseInt(d3.select('#div_main').style('width'), 10)
  Svg.attr("width", currentWidth - 20)

  // Update the X scale 
  scales["x"].range([ 50, currentWidth]);
  heatmap.attrs({"x": (d) => ratio*scales.x(d.coords[0]),
    "width": ratio*scales.x(ux[1]) - ratio*scales.x(ux[0])
  })

  points.attrs({"cx": (d) => ratio*scales.x(d.coords[0])})
  // allows interactivity with points
  Svg.selectAll("line").remove()

}


//
// Initialize the chart
drawChart()

// Add an event listener that run the function when dimension change
window.addEventListener('resize', drawChart );


//arrow
Svg.append("svg:defs").append("svg:marker")
    .attr("id", "triangle")
    .attr("refX", 6)
    .attr("refY", 6)
    .attr("markerWidth", 30)
    .attr("markerHeight", 30)
    .attr("markerUnits","userSpaceOnUse")
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M 0 0 12 6 0 12 3 6")
    .style("fill", "black");
    


d3.selectAll("rect")
    .on("mouseover", handleMouseOver)
    .on("mouseout", handleMouseOut)
    .on("click", handleClick);
		    

function handleClick(d, i) {
   var coords = d3.mouse(this);
   var coords_o = d.coords;
   
    Svg.append("line")
        .attr("x1",  ratio*scales.x(min_x + (max_x - min_x)/2))
        .attr("y1", scales.y(min_y + (max_y - min_y)/2))
        .attr("x2", coords[0])
        .attr("y2", coords[1])
	.attr("coords", coords_o)
        .attr("stroke-width", 2)
        .attr("stroke", "red")
        .attr("id", "#p" + "-" + i)
        .attr("marker-end", "url(#triangle)");
     data.push([coords_o[0].toPrecision(4),coords_o[1].toPrecision(4)])
     var tRows = t.selectAll('tr')
       .data(data)
       .enter()
       .append('tr');
    
     tRows
       .selectAll('td')
       .data(function(d) {
         return d3.values(d);
       })
       .enter()
       .append('td')
       .html(function(d) {
         return d;
       });


  var selected = points_data.filter(
    d => ratio*scales.x(min_x + (max_x - min_x)/2) <= ratio*scales.x(d.coords[0]) && ratio*scales.x(d.coords[0]) < coords[0] &&
      scales.y(min_y + (max_y - min_y)/2) <= scales.y(d.coords[1]) && scales.y(d.coords[1]) < coords[1]
  ).slice(0,8);


  patch_sel = update_images(selected.map((d) => d.patch_path), 'patch',"es", 0);
  patch_sel.exit();

 }

function handleMouseOver(d, i) {  // Add interactivity
    var coords = d.coords;
    Svg.append("line")
        .attr("x1",  ratio*scales.x(min_x + (max_x - min_x)/2))
        .attr("y1", scales.y(min_y + (max_y - min_y)/2))
        .attr("x2", ratio*scales.x(coords[0]))
        .attr("y2", scales.y(coords[1]))
	.attr("coords",coords)
        .attr("stroke-width", 2)
        .attr("stroke", "black")
        .attr("id", "#t" + "-" + i)
        .attr("marker-end", "url(#triangle)");

}


function handleMouseOut(d, i) {
      var element = document.getElementById("#t" + "-" + i);
      element.parentNode.removeChild(element);
}


// On Click, we want to add data to the array and chart

var t = d3.select('body').append("div").attr("class","table-responsive-sm").attr("style","width:400px;").append('table').attr("class","table");
var data = [["1st-pc","2nd-pc"]];
 t.append('tr')
   .selectAll('th')
   .data(data[0])
   .enter()
   .append('th')
   .text(function(d) {
     return d;
   });

 var tRows = t.selectAll('tr')
   .data(data)
   .enter()
   .append('tr');

 tRows
   .selectAll('td')
   .data(function(d) {
     return d3.values(d);
   })
   .enter()
   .append('td')
   .html(function(d) {
     return d;
   });
