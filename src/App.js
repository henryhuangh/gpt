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
      </header>
      <ChatBot />
    </div>
  );
}

export default App;
