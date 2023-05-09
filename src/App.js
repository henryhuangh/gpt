import logo from "./logo.svg";
import "./App.css";
import ChatBot from "./components/chatbot_widget";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt="logo" className="mx-5" />
        <p>
          Chatbot
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
      <ChatBot />
    </div>
  );
}

export default App;
