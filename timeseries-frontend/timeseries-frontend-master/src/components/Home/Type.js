import React from "react";
import Typewriter from "typewriter-effect";

function Type() {
    return (
        <Typewriter
            options={{
                strings: [
                    "Pick a model. Upload your data. Watch it predict",
                    "One platform. Many models. Infinite insights",
                    "Let your data speak — our models listen",
                    "Prediction has never been this intuitive",
                ],
                autoStart: true,
                loop: true,
                deleteSpeed: 50,
            }}
        />
    );
}

export default Type;
