// Our Region of Interest is California (region around Fresno)
var point = ee.Geometry.Point([-119.7871, 36.7378]);
var roi = ee.Geometry(point).buffer(15000); 

var dataset = ee.Image('NASA/NASADEM_HGT/001');
var elevation = dataset.select('elevation').clip(roi);

Export.image.toDrive({
  image: elevation,
  description: 'NASADEM_Elevation_Fresno_10m',
  folder: 'PINN_LST_Project_Data', 
  scale: 10,                       // Forces 10m grid alignment
  region: roi,
  maxPixels: 1e10                  
});

Map.setCenter(-119.7871, 36.7378, 11);
Map.addLayer(elevation, {min: 50, max: 150, palette: ['blue', 'green', 'red']}, 'Fresno Elevation');