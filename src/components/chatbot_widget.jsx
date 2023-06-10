import React, { Component, useEffect, useState } from "react";
import PropTypes from "prop-types";
import ChatBot, { Loading } from "react-simple-chatbot";
import { v4 as uuidv4 } from "uuid";

const ChatBotWidget = () => {
  const [chatID, setChatID] = useState(uuidv4());
  const Post = (props) => {
    const [loading, setLoading] = useState(true);
    const [result, setResult] = useState("");
    const [trigger, setTrigger] = useState(false);

    const triggetNext = () => {
      setTrigger(true);
      props.triggerNextStep();
    };
    useEffect(() => {
      async function fetchResponse() {
        const query = props.steps.search.value;

        const response = await fetch("/chat", {
          method: "POST",
          keepalive: true,
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({
            response: query,
            chat_id: props.chat_id === null ? undefined : props.chat_id,
          }),
        });
        try {
          const reader = response.body.getReader();
          if (!response.ok) {
            throw new Error("HTTP status " + response.status);
          }
          setLoading(false);
          var resultText = "";
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            resultText += new TextDecoder().decode(value);
            setResult(resultText);
          }
          // const data = await response.json();
          // console.log(data);

          // setResult(data["response"]);
          triggetNext();
        } catch (err) {
          console.log("Error: ", err);
          setLoading(false);
          setResult("Sorry, an error occured");
        }
      }
      fetchResponse();
    }, []);

    return <div>{loading ? <Loading /> : result}</div>;
  };

  return (
    <ChatBot
      steps={[
        {
          id: "1",
          message: "Ask me anything?",
          trigger: "search",
        },
        {
          id: "search",
          user: true,
          validator: (value) => {
            if (value == "") {
              return "message should not be blank";
            }
            return true;
          },
          trigger: "result",
        },
        {
          id: "result",
          component: <Post chat_id={chatID} setChatID={setChatID} />,
          asMessage: true,
          waitAction: true,
          trigger: "search",
        },
      ]}
      floating
      width={"600px"}
      userDelay={200}
      customDelay={20}
      headerTitle={"Ask me anything"}
      height={"700px"}
    />
  );
};

export default ChatBotWidget;
