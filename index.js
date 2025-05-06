
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

  var x = new Float32Array(1, 3)
  var x = [];
  x[0] = parseFloat(document.getElementById('box1').value);
  x[1] = parseFloat(document.getElementById('box2').value);
  x[2] = parseFloat(document.getElementById('box3').value);

  for (let i = 0; i < x.length; i++) {
    x[i] = (x[i] - mean[i]) / (std[i] + 0.0001);
  }

  let tensorX = new onnx.Tensor(x, 'float32', [1, 3]);


  try {
    let session = new onnx.InferenceSession();
    await session.loadModel("DLnet_WeatherData.onnx");

   
    let outputMap = await session.run([tensorX]);
    //let inputMap = await session.run(['Chicago', 'Dallas', 'Los Angeles', 'New York', 'Philadelphia']) 
    let outputData = outputMap.get('output1');
    //let inputData = inputMap.get('input1');

    let predictions = document.getElementById('predictions');
    predictions.innerHTML = `
      <hr> Predicted Location: <br/>
      <table>
        <tr>
          <td>  City: ${inputData.data[0].toFixed(2)} </td>
          <td id="td0"> ${outputData.data[0].toFixed(2)} </td>
        </tr>
      </table>`;
  } catch (err) {
    console.error(err);
    document.getElementById("predictions").innerHTML = "Error running model: " + err.message;
  }
}