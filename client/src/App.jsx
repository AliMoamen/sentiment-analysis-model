import { useState } from "react";
import axios from "axios";
import "./App.css";

const App = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text) {
      alert("Please enter some text.");
      return;
    }

    setLoading(true);

    try {
      // Send POST request to Flask API
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        text,
      });
      setResult(response.data);
    } catch (error) {
      console.error("There was an error fetching the prediction:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Sentiment Analysis</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={handleTextChange}
          rows="4"
          cols="50"
          placeholder="Enter text here"
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Loading..." : "Analyze Sentiment"}
        </button>
      </form>

      {result && (
        <div className="result">
          <h2>Prediction Results:</h2>
          <p>
            <strong>Sentiment:</strong> {result.predicted_label}
          </p>
          <p>
            <strong>Confidence:</strong> {result.confidence.toFixed(2)}
          </p>
          <p>
            <strong>Class Probabilities:</strong>
          </p>
          <ul>
            {Object.entries(result.class_probabilities).map(([label, prob]) => (
              <li key={label}>
                {label}: {prob.toFixed(2)}
              </li>
            ))}
          </ul>
          <p>
            <strong>Uncertainty Score:</strong>{" "}
            {result.uncertainty_score.toFixed(2)}
          </p>
        </div>
      )}
    </div>
  );
};

export default App;
