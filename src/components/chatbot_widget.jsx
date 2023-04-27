import React, { Component, useEffect, useState } from "react";
import PropTypes from "prop-types";
import ChatBot, { Loading } from "react-simple-chatbot";

const ChatBotWidget = () => {
  const [chatID, setChatID] = useState(null);
  class Post extends Component {
    constructor(props) {
      super(props);

      this.state = {
        loading: true,
        result: "",
        trigger: false,
      };

      this.triggetNext = this.triggetNext.bind(this);
    }

    triggetNext() {
      this.setState({ trigger: true }, () => {
        this.props.triggerNextStep();
      });
    }

    async componentWillMount() {
      const self = this;
      const { steps, chatID } = this.props;
      const query = steps.search.value;
      const response = await fetch("/chat", {
        method: "POST",
        body: JSON.stringify({
          response: query,
          chat_id: chatID === null ? undefined : chatID,
        }),
        headers: { "Content-Type": "application/json" },
      });

      try {
        const data = await response.json();
        self.triggetNext();
        if (data["response"]) {
          self.setState({ loading: false, result: data["response"] });
          setChatID(data["chat_id"]);
        }
      } catch (err) {
        console.log("Error: ", err);
        self.setState({ loading: false, result: "Sorry, an error occured" });
      }
    }

    render() {
      const { trigger, loading, result } = this.state;

      return <div>{loading ? <Loading /> : result}</div>;
    }
  }

  Post.propTypes = {
    steps: PropTypes.object,
    triggerNextStep: PropTypes.func,
  };

  Post.defaultProps = {
    steps: undefined,
    triggerNextStep: undefined,
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
          component: <Post chatID={chatID} />,
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
