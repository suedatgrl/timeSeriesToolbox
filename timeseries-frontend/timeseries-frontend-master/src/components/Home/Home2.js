import React from "react";
import {Col, Container, Row} from "react-bootstrap";
import {AiFillGithub, AiFillInstagram, AiOutlineTwitter,} from "react-icons/ai";
import {FaLinkedinIn} from "react-icons/fa";
import Particle from "../Particle";

function Home2() {
    return (
        <Container fluid className="home-section" id="home">
            <Particle/>
            <br/>
            <h3 style={{fontSize: "2em"}}>
                <span className="purple" style={{marginRight: "5px"}}> Yusuf Emre Yıldırım </span>
                |||||
                <span className="white" style={{marginRight: "5px"}}> Kutay Kaan Koçak </span>
                |||||
                <span className="purple" style={{marginRight: "5px"}}> Sibel Sıla Aslan </span>
                |||||
                <span className="white" style={{marginRight: "5px"}}> Latife Süeda Tuğrul </span>
                |||||
                <span className="purple" style={{marginRight: "5px"}}> Sude Korkmaz </span>
                <br/>
            </h3>
            <Container className="home-content">
                <Row>
                    <Col md={12} className="home-about-description">
                        <h1 style={{fontSize: "2.6em"}}>
                            👋 LET US <span className="purple"> INTRODUCE </span> OURSELVES
                        </h1>
                        <p className="home-about-body">
                            We are a team of 5 engineers passionate about Time Series Modeling and Forecasting.
                            <br/>
                            <br/>Together, we’ve built TimeSeries Lab — a dynamic platform that brings together powerful
                            models like
                            .
                            <i>
                                <b className="purple"> LSTM, ARIMA, Prophet,
                                    Sarima, XGBoost, Random Forest, CNN, Rocket, InceptionTime, Shapelet Transform Classifier </b>
                            </i>
                            <br/>
                            <br/>
                            <br/>
                            Welcome to the TimeSeries Lab Team
                            <i>
                                <br/>
                                <b className="purple">
                                    {" "}
                                    Where Time Meets Intelligence.
                                </b>
                            </i>
                        </p>
                    </Col>
                </Row>
            </Container>
        </Container>
    );
}

export default Home2;
