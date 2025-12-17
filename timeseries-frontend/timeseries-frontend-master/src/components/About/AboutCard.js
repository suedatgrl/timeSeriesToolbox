import React from "react";
import Card from "react-bootstrap/Card";
import {ImPointRight} from "react-icons/im";

function AboutCard() {
    return (
        <Card className="quote-card-view">
            <Card.Body>
                <blockquote className="blockquote mb-0">
                    <p style={{textAlign: "justify"}}>
                        Hi everyone! We're <span className="purple">TimeSeries Forge Team </span>
                        a group of five passionate students from
                        <span className="purple"> Ankara University, Department of Computer Engineering </span>
                        specializing in time series forecasting and intelligent model design.
                        <br/>
                    </p>
                    <br/>

                    <ul style={{display: "flex", justifyContent: "space-around"}}>
                        <li className="about-activity">
                            <p style={{textAlign: "center"}}>Prediction</p>
                            <hr/>
                            <span className="purple" style={{display: "block"}}> Arima</span>
                            <span className="purple" style={{display: "block"}}> Prophet</span>
                            <span className="purple" style={{display: "block"}}> XgBoost</span>
                            <span className="purple" style={{display: "block"}}> LSTM</span>
                            <span className="purple" style={{display: "block"}}> Sarima</span>
                        </li>
                        <li className="about-activity">
                            <p style={{textAlign: "center"}}>Classification</p>
                            <hr/>
                            <span className="purple" style={{display: "block"}}> Random Forest</span>
                            <span className="purple" style={{display: "block"}}> CNN</span>
                            <span className="purple" style={{display: "block"}}> Rocket</span>
                            <span className="purple" style={{display: "block"}}> InceptionTime</span>
                            <span className="purple" style={{display: "block"}}> Shapelet Transform Classifier</span>
                        </li>
                    </ul>
                    <br/>

                    <p style={{color: "rgb(155 126 172)"}}>
                        Where Time Meets Intelligence
                    </p>
                    <footer className="blockquote-footer">TimeSeries Forge Team</footer>
                </blockquote>
            </Card.Body>
        </Card>
    );
}

export default AboutCard;
