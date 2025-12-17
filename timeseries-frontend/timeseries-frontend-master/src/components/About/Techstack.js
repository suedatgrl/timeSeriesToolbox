import React from "react";
import {Col, Row} from "react-bootstrap";
import {CgCPlusPlus} from "react-icons/cg";
import {
    DiJavascript1,
    DiReact,
    DiNodejs,
    DiMongodb,
    DiPython,
    DiGit,
    DiJava,
} from "react-icons/di";
import {
    SiRedis,
    SiFirebase,
    SiNextdotjs,
    SiSolidity,
    SiPostgresql,
    SiFlask,
    SiPython,
    SiReact,
    SiTensorflow,
    SiMui,
    SiTypescript,
    SiPandas,
    SiNumpy,
    SiScikitlearn,
} from "react-icons/si";
import {
    GiNetworkBars
} from "react-icons/gi";
import {TbBrandGolang, TbWaveSawTool} from "react-icons/tb";
import {FaBrain, FaChartLine} from "react-icons/fa";

function Techstack() {
    return (
        <Row style={{justifyContent: "center", paddingBottom: "50px"}}>
            <Col xs={4} md={2} className="tech-icons">
                <SiReact/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiTypescript/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <DiJavascript1/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiPython/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiFlask/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiPandas/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiNumpy/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiScikitlearn/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <DiGit/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiTensorflow/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <FaBrain/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <FaChartLine/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <GiNetworkBars/>
            </Col>
            <Col xs={4} md={2} className="tech-icons">
                <SiMui />
            </Col>
            {/*<Col xs={4} md={2} className="tech-icons">*/}
            {/*    <MdAutoGraph/>*/}
            {/*</Col>*/}
        </Row>
    );
}

export default Techstack;
