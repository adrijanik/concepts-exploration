/**
 * Experiment with Interactive Concepts Discovery
 */

var Svg = d3.select("#div_main")
  .append("svg")
  .attr("id", "svg")
  .attr("height", 1000)

var index = 100;

var min_g0 = -2935.449393712765;
var min_g1 = -765.6401511302116;
var max_g0 = -2935.4493522859743;
var max_g1 = -765.6401444491758;
var min_x = -3054.9271780362033;
var max_x = 1953.187179198643;
var min_y = -2101.9897178164883;
var max_y = 2556.7949219460525;


var scales = {
  "x": d3.scaleLinear()
    .domain([min_x, max_x])
    .range([50, 2000]),
  "y": d3.scaleLinear()
    .domain([min_y, max_y])
    .range([30, 865]),
  "color": d3.scaleSequential(d3.interpolateRdBu)
    .domain([0,1])
};

var ratio = 1;


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

const range = (start, stop, step = 1) => Array(Math.ceil((stop - start) / step)).fill(start).map((x, y) => x + y * step);

var legendLinear = d3.legendColor()
  .shapeWidth(35)
  .shapeHeight(30)
  .cells([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
  .orient('horizontal')
  .scale(scales.color);

legend.select(".legendLinear")
  .call(legendLinear);


d3.json("/data/classes.json").then(function(data) {
  var checklist = d3.select('#checklist').selectAll("input")
    .data(data)
    .enter()
    .append('tr')
    .append('td')
    .append('label')
        .text(function(d) { return d.class.split(',')[0]; })
    .append("input")
    .attrs({ "type":"checkbox", "name":(d) => d.id, "value":(d) => d.class, "class":"check1"});

});

// draw the scatterplot points
var points = points.selectAll("circle")
  .data(points_data)
  .enter()
  .append("circle")
  .attrs({
    "cy": (d) => scales.y(d.coords[1]),
    "r": 4,
    "fill": 'white',
    "stroke": "black",
    "stroke-width": 1,
    "id": (d) => d.id
  });


// draw the background
var ux = [... new Set(hm_data.map((d) => d.coords[0]))],
    uy = [... new Set(hm_data.map((d) => d.coords[1]))];

var heatmap = heatmap.selectAll("rect")
  .data(hm_data).enter()
  .append("rect")
  .attrs({
    "y": (d) => scales.y(d.coords[1]),
    "height": scales.y(uy[1]) - scales.y(uy[0]),
    "fill": (d) => scales.color(d.iou),
    "opacity": 0
  });


var values0 = points_data.map(function(dict){
  return dict['coords'][0];
}); 
var values1 = points_data.map(function(dict){
  return dict['coords'][1];
}); 


function selectClass() {
   	
    var checkedValue = null; 
    var inputElements = document.querySelectorAll('.check1');

    for(var i=0; inputElements[i]; ++i){
          if(inputElements[i].checked){
               checkedValue = inputElements[i].value;
               break;
          }
    }

    Svg.selectAll("circle")
    .filter(function(d) { return d.class == checkedValue })
    .attr("fill", "gray" );


    return false;
}


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
//    var x1 =  ratio*scales.x(min_x + (max_x - min_x)/2);
//    var y1 = scales.y(min_y + (max_y - min_y)/2);
   var x1 =  ratio*scales.x(0);
   var y1 = scales.y(0);


    Svg.append("line")
        .attr("x1",  x1)
        .attr("y1", y1)
        .attr("x2", coords[0])
        .attr("y2", coords[1])
	.attr("coords", coords_o)
        .attr("stroke-width", 2)
        .attr("stroke", "red")
        .attr("id", "#p" + "-" + i)
        .attr("marker-end", "url(#triangle)");
//     data.push([coords_o[0].toPrecision(4),coords_o[1].toPrecision(4)])
//     var tRows = t.selectAll('tr')
//       .data(data)
//       .enter()
//       .append('tr');
//    
//     tRows
//       .selectAll('td')
//       .data(function(d) {
//         return d3.values(d);
//       })
//       .enter()
//       .append('td')
//       .html(function(d) {
//         return d;
//       });

 }

function handleMouseOver(d, i) {  // Add interactivity
    var coords = d.coords;
//    var x1 =  ratio*scales.x(min_x + (max_x - min_x)/2);
//    var y1 = scales.y(min_y + (max_y - min_y)/2);
    var x1 =  ratio*scales.x(0);
    var y1 = scales.y(0);

    var x2 = ratio*scales.x(coords[0]);
    var y2 = scales.y(coords[1]);
    var x3 = (y1-y2)/(x1-x2);
    var y3 = -1;
    var mag = Math.sqrt(Math.pow(x3,2)+Math.pow(y3,2));
    var len = 400/((mag == 0) ? 1 : mag);

    var x4 = x1 + x3*len;
    var y4 = y1 + y3*len;
    Svg.append("line")
        .attr("x1",  x1)
        .attr("y1", y1)
        .attr("x2", x2)
        .attr("y2", y2)
	.attr("coords",coords)
        .attr("stroke-width", 2)
        .attr("stroke", "black")
        .attr("id", "#t" + "-" + i)
        .attr("marker-end", "url(#triangle)");

     Svg.append("line")
        .attr("x1", -1*(x4-x1) + x1)
        .attr("y1", -1*(y4-y1) + y1)
        .attr("x2", x4)
        .attr("y2", y4)
	.attr("coords",coords)
        .attr("stroke-width", 2)
        .attr("stroke", "gray")
        .style("stroke-dasharray", ("3, 3"))	
        .attr("id", "#t_ort" + "-" + i);

     var angle =  document.getElementById("valBox").innerHTML * Math.PI/180;

     var right_x = x1 + Math.cos(angle)*(x2-x1)-Math.sin(angle)*(y2-y1);
     var left_x = x1 + Math.cos(-1*angle)*(x2-x1)-Math.sin(-1*angle)*(y2-y1);
     var right_y = y1 + Math.sin(angle)*(x2-x1)+Math.cos(angle)*(y2-y1);
     var left_y = y1 + Math.sin(-1*angle)*(x2-x1)+Math.cos(-1*angle)*(y2-y1);

     Svg.append("line")
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", right_x)
        .attr("y2", right_y)
	.attr("coords",coords)
        .attr("stroke-width", 2)
        .attr("stroke", "gray")
        .style("stroke-dasharray", ("3, 3"))	
        .attr("id", "#t_ang" + "-" + i);
 
     Svg.append("line")
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", left_x)
        .attr("y2", left_y)
	.attr("coords",coords)
        .attr("stroke-width", 2)
        .attr("stroke", "gray")
        .style("stroke-dasharray", ("3, 3"))	
        .attr("id", "#t_ang2" + "-" + i);
 
  var right_a = (y1 - right_y)/(x1 - right_x);
  var right_b = y1 - right_a*x1;

  var left_a = (y1 - left_y)/(x1 - left_x);
  var left_b = y1 - left_a*x1;
  var R_2 = Math.pow(left_x-x1,2)+Math.pow(left_y-y1,2);

  var selected = points_data.filter(
    d => Math.pow((ratio*scales.x(d.coords[0]) - x1),2) + Math.pow((scales.y(d.coords[1]) - y1),2) < R_2 && (
         right_a * ratio*scales.x(d.coords[0]) + right_b >  scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b <= scales.y(d.coords[1]) && right_a < 0 && left_a < 0 && ratio*scales.x(d.coords[0]) > x1 && left_x < right_x ||
         right_a * ratio*scales.x(d.coords[0]) + right_b <= scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b >  scales.y(d.coords[1]) && right_a > 0 && left_a > 0 && left_x < right_x ||
         right_a * ratio*scales.x(d.coords[0]) + right_b <= scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b <= scales.y(d.coords[1]) && right_a < 0 && left_a > 0 && scales.y(d.coords[1]) > y1 && left_x > right_x || 
         right_a * ratio*scales.x(d.coords[0]) + right_b >  scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b >  scales.y(d.coords[1]) && right_a < 0 && left_a > 0 && scales.y(d.coords[1]) < y1 && left_x < right_x || 
         right_a * ratio*scales.x(d.coords[0]) + right_b >  scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b <= scales.y(d.coords[1]) && right_a > 0 && left_a > 0 && ratio*scales.x(d.coords[0]) > x1 && left_x > right_x ||
         right_a * ratio*scales.x(d.coords[0]) + right_b <= scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b >  scales.y(d.coords[1]) && right_a < 0 && left_a < 0 && left_x > right_x ||
         right_a * ratio*scales.x(d.coords[0]) + right_b <= scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b >  scales.y(d.coords[1]) && right_a > 0 && left_a < 0 && left_x < x1   || 
         right_a * ratio*scales.x(d.coords[0]) + right_b >  scales.y(d.coords[1]) && left_a * ratio*scales.x(d.coords[0]) + left_b <= scales.y(d.coords[1]) && right_a > 0 && left_a < 0 && left_x > x1 )
  );


  var mapped = selected.map((k) => k.id);
  Svg.selectAll("circle")
  .filter(function(d) { return mapped.includes(d.id) })
  .attr("fill", "black");

   var checkedValue = null; 
   var inputElements = document.querySelectorAll('.check1');

   for(var i=0; inputElements[i]; ++i){
         if(inputElements[i].checked){
              checkedValue = inputElements[i].value;
              break;
         }
   }

  function stand(x, min, max) {
    return (x-min)/(max-min);
  }
  
  Svg.selectAll("circle")
  .filter(function(d) { return d.class == checkedValue })
  .attr("fill", (d) =>  scales.color(math.multiply([stand(d.coords[0],min_x, max_x),stand(d.coords[1],min_y, max_y)],[stand(d.gradient[0],min_g0, max_g0),stand(d.gradient[1],min_g1, max_g1)])))
  .attr("r", 10); 


  patch_sel = update_images(selected.map((d) => d.patch_path), 'patch',"es", 0);
  patch_sel.exit();


}


function handleMouseOut(d, i) {
      var element = document.getElementById("#t" + "-" + i);
      element.parentNode.removeChild(element);
      var element = document.getElementById("#t_ort" + "-" + i);
      element.parentNode.removeChild(element);
      var element = document.getElementById("#t_ang" + "-" + i);
      element.parentNode.removeChild(element);
      var element = document.getElementById("#t_ang2" + "-" + i);
      element.parentNode.removeChild(element);
      var element = document.getElementById("grid");
      element.parentNode.removeChild(element);

      var checkedValue = null; 
      var inputElements = document.querySelectorAll('.check1');
  
      for(var i=0; inputElements[i]; ++i){
            if(inputElements[i].checked){
                 checkedValue = inputElements[i].value;
                 break;
            }
      }


      Svg.selectAll("circle")
      .filter(function(d) { return d.class != checkedValue })
      .attr("fill", "white")
      .attr("r", 4);

}


// On Click add data to the array and chart

//var t = d3.select('body').append("div").attr("class","table-responsive-sm").attr("style","width:400px;").append('table').attr("class","table");
//var data = [["1st-pc","2nd-pc"]];
// t.append('tr')
//   .selectAll('th')
//   .data(data[0])
//   .enter()
//   .append('th')
//   .text(function(d) {
//     return d;
//   });
//
// var tRows = t.selectAll('tr')
//   .data(data)
//   .enter()
//   .append('tr');
//
// tRows
//   .selectAll('td')
//   .data(function(d) {
//     return d3.values(d);
//   })
//   .enter()
//   .append('td')
//   .html(function(d) {
//     return d;
//   });
