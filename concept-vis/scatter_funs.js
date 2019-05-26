/**
 * Functions for drawing the scatterplot
 */


function update_images(im_ids, name, plural, offset) {
  currentWidth = parseInt(d3.select('#div_main').style('width'), 10)
  var initial = scales.x(Math.max.apply(null, values0)) + ((currentWidth  -  (currentWidth))/3 - 115)

  var Grid = d3.select('#col2')
          .append('grid')
          .attr('id', 'grid')
          .attr('class','grid');


  var imgs = Grid
        .selectAll("div")
        .data(im_ids)
        .enter()
        .append("div")
        .attr("class", "patch")
	.attr("height",200)
	.attr("width",200)
      ;

  imgs
     .style("background-image", function(d){
        return 'url("'+d+'")';
     })
  ;
  return Grid
}



