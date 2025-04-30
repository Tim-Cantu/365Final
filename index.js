
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

  

  let x = [];

  x[0] = parseFloat(document.getElementById('box1').value);
  x[1] = parseFloat(document.getElementById('box2').value);
  x[2] = parseFloat(document.getElementById('box3').value);

  for (let i = 0; i < x.length; i++) {
      x[i] = (x[i] - mean[i]) / (std[i] + .0001); // same epsilon
  }

  // Create tensor
  let tensorX = new onnx.Tensor(new Float32Array(x), 'float32', [1, 3]);

  // Load model
  let session = new onnx.InferenceSession();
  await session.loadModel("./DLnet_WeatherData.onnx");

  // Run model
  let outputMap = await session.run([tensorX]);
  let outputData = outputMap.get('output1');

  // Apply softmax for probability
  function softmax(arr) {
      const max = Math.max(...arr);
      const exps = arr.map(x => Math.exp(x - max));
      const sum = exps.reduce((a, b) => a + b);
      return exps.map(x => x / sum);
  }

  let probs = softmax(Array.from(outputData));
  let maxIndex = probs.indexOf(Math.max(...probs));


    // Display results
    let predictions = document.getElementById('predictions');

    let city = ["Chicago", "Dallas", "Los Angeles", "New York", "Philadelphia"];

    predictions.innerHTML = `
        <hr> Predicted Location: <b>${labels[maxIndex]}</b><br/>
        Probabilities: <br/>
        <table>
            ${probs.map((p, i) => `
                <tr><td>${labels[i]}</td><td>${(p * 100).toFixed(2)}%</td></tr>
            `).join("")}
        </table>
    `;
}