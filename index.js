
/*async function runExample() {

    var x = new Float32Array( 1, 3 )

    var x = [];

     x[0] = document.getElementById('box1').value;
     x[1] = document.getElementById('box2').value;
     x[2] = document.getElementById('box3').value;
     

    let tensorX = new onnx.Tensor(x, 'float32', [1, 3]);

    let session = new onnx.InferenceSession();

    await session.loadModel("./DLnet_WeatherData.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

   let predictions = document.getElementById('predictions');

  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Location based on Weather   </td>
       <td id="td0">  ${outputData.data[0].toFixed(2)}  </td>
     </tr>
  </table>`;
    


}*/



async function runExample() {
  
  const mean = [14.2834, 57.8935, 14.8043];  
  const std  = [14.4519, 18.0324,  8.5738];   

  // Get inputs and normalize them

  var x = new Float32Array(1, 3)

  x[0] = parseFloat(document.getElementById('box1').value);
  x[1] = parseFloat(document.getElementById('box2').value);
  x[2] = parseFloat(document.getElementById('box3').value);

  for (let i = 0; i < x.length; i++) {
      x[i] = (x[i] - mean[i]) / (std[i] + .0001); // same epsilon
  }

  // Create tensor
  let tensorX = new onnx.Tensor(x, 'float32', [1, 3]);

  // Load model
  let session = new onnx.InferenceSession();
  await session.loadModel("./DLnet_WeatherData.onnx");

  // Run model
  let outputMap = await session.run({ input: tensorX });
  let outputData = outputMap.get('output1');

  


    // Display results
    let predictions = document.getElementById('predictions');


    predictions.innerHTML = `
        <hr> Predicted Location: <br/>
        Probabilities: <br/>
        <table>
            <tr>
                <td> Rating of City </td>
                <td id= "td0"> ${outputData.data[0].toFixed(2)} </td>
            </tr>
        </table>`;
}