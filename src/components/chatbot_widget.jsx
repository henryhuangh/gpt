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

        const response = await fetch("http://ryerson.xyz/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({
            response: query,
            chat_id: props.chat_id === null ? undefined : props.chat_id,
          }),
        });
        console.log(response);
        try {
          const data = await response.json();
          console.log(data);
          setLoading(false);
          setResult(data["response"]);
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
              return "query should not be blank";
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
      headerTitle={"Ask me anything"}
    />
  );
};

export default ChatBotWidget;
