import React, {useState} from "react";
import Particle from "../Particle";

function Projects() {
    const [file, setFile] = useState(null);
    const [filename, setFilename] = useState('');
    const [columns, setColumns] = useState([]);
    const [previewData, setPreviewData] = useState([]);
    const [targetColumn, setTargetColumn] = useState('');
    const [modelType, setModelType] = useState('cnn');
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState('');

    // CNN model parameters
    const [cnnSequenceLength, setCnnSequenceLength] = useState(10);
    const [cnnNumFilters, setCnnNumFilters] = useState(64);
    const [cnnKernelSize, setCnnKernelSize] = useState(3);
    const [cnnDenseUnits, setCnnDenseUnits] = useState(64);
    const [cnnEpochs, setCnnEpochs] = useState(50);

    // LSTM model parameters
    const [lstmUnits, setLstmUnits] = useState(50);
    const [lstmDenseUnits, setLstmDenseUnits] = useState(64);
    const [lstmEpochs, setLstmEpochs] = useState(50);

    // ARIMA model parameters
    const [arimaPValue, setArimaPValue] = useState(1);
    const [arimaDValue, setArimaDValue] = useState(1);
    const [arimaQValue, setArimaQValue] = useState(1);

    // Prophet model parameters
    const [prophetSeasonalityMode, setProphetSeasonalityMode] = useState('additive');
    const [prophetChangePointPrior, setProphetChangePointPrior] = useState(0.05);
    const [prophetSeasonalityPrior, setProphetSeasonalityPrior] = useState(10);

    // XGBoost model parameters
    const [xgboostMaxDepth, setXgboostMaxDepth] = useState(6);
    const [xgboostLearningRate, setXgboostLearningRate] = useState(0.1);
    const [xgboostNEstimators, setXgboostNEstimators] = useState(100);

    // Transformer model parameters
    const [transformerNumHeads, setTransformerNumHeads] = useState(8);
    const [transformerNumEncoderLayers, setTransformerNumEncoderLayers] = useState(4);
    const [transformerDropoutRate, setTransformerDropoutRate] = useState(0.1);
    const [transformerDimModel, setTransformerDimModel] = useState(64);
    const [transformerEpochs, setTransformerEpochs] = useState(50);

    // SARIMA model parameters
    const [sarimaP, setSarimaP] = useState(1);
    const [sarimaD, setSarimaD] = useState(0);
    const [sarimaQ, setSarimaQ] = useState(1);
    const [sarimaSP, setSarimaSP] = useState(1);
    const [sarimaSD, setSarimaSD] = useState(0);
    const [sarimaSQ, setSarimaSQ] = useState(1);
    const [sarimaSeasonalPeriod, setSarimaSeasonalPeriod] = useState(12);

    // Random Forest model parameters
    const [rfEstimators, setRfEstimators] = useState(100);
    const [rfMaxDepth, setRfMaxDepth] = useState(null);
    const [rfMinSamplesSplit, setRfMinSamplesSplit] = useState(2);
    const [rfMinSamplesLeaf, setRfMinSamplesLeaf] = useState(1);

    // ROCKET model parameters
    const [rocketNumKernels, setRocketNumKernels] = useState(2000);
    const [rocketWindowSize, setRocketWindowSize] = useState(30);
    const [rocketStride, setRocketStride] = useState(1);

    // InceptionTime model parameters
    const [inceptionNbFilters, setInceptionNbFilters] = useState(32);
    const [inceptionUseResidual, setInceptionUseResidual] = useState(true);
    const [inceptionUseBottleneck, setInceptionUseBottleneck] = useState(true);
    const [inceptionDepth, setInceptionDepth] = useState(6);
    const [inceptionKernelSize, setInceptionKernelSize] = useState(3);
    const [inceptionEpochs, setInceptionEpochs] = useState(50);

    // Shapelet Transform model parameters
    const [shapeletNShapelets, setShapeletNShapelets] = useState(100);
    const [shapeletLengths, setShapeletLengths] = useState([0.1, 0.2, 0.3]);
    const [shapeletSequenceLength, setShapeletSequenceLength] = useState(10);


    const handleFileChange = (e) => {
        if (e.target.files[0]) {
            setFile(e.target.files[0]);
            setError('');
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a file first.');
            return;
        }

        setIsLoading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:5000/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('File upload failed');
            }

            const data = await response.json();

            setFilename(data.filename);
            setColumns(data.columns);
            setTargetColumn(data.columns[0]);

            // Get data preview
            const previewResponse = await fetch(`http://localhost:5000/api/preview/${data.filename}`);

            if (!previewResponse.ok) {
                throw new Error('Failed to get data preview');
            }
            const previewData = await previewResponse.json();

            setPreviewData(previewData.head);

        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleModelTrain = async () => {
        if (!filename || !targetColumn) {
            setError('Please upload a file and select a target column first.');
            return;
        }

        setIsLoading(true);
        setError('');

        try {
            const modelParams = {
                filename,
                targetColumn,
                modelType,
                params: {},
            };

            switch (modelType) {
                case 'cnn':
                    modelParams.params = {
                        sequenceLength: parseInt(cnnSequenceLength),
                        numFilters: parseInt(cnnNumFilters),
                        kernelSize: parseInt(cnnKernelSize),
                        denseUnits: parseInt(cnnDenseUnits),
                        epochs: parseInt(cnnEpochs),
                    };
                    break;
                case 'lstm':
                    modelParams.params = {
                        lstmUnits: parseInt(lstmUnits),
                        denseUnits: parseInt(lstmDenseUnits),
                        epochs: parseInt(lstmEpochs),
                    };
                    break;
                case 'arima':
                    modelParams.params = {
                        p: parseInt(arimaPValue),
                        d: parseInt(arimaDValue),
                        q: parseInt(arimaQValue),
                    };
                    break;
                case 'prophet':
                    modelParams.params = {
                        seasonalityMode: prophetSeasonalityMode,
                        changePointPrior: parseFloat(prophetChangePointPrior),
                        seasonalityPrior: parseFloat(prophetSeasonalityPrior),
                    };
                    break;
                case 'xgboost':
                    modelParams.params = {
                        maxDepth: parseInt(xgboostMaxDepth),
                        learningRate: parseFloat(xgboostLearningRate),
                        nEstimators: parseInt(xgboostNEstimators),
                        sequenceLength: parseInt(cnnSequenceLength), // XGBoost için sequenceLength gerekebilir (lag oluşturma)
                    };
                    break;
                case 'transformer':
                    modelParams.params = {
                        numHeads: parseInt(transformerNumHeads),
                        numEncoderLayers: parseInt(transformerNumEncoderLayers),
                        dropoutRate: parseFloat(transformerDropoutRate),
                        dimModel: parseInt(transformerDimModel),
                        epochs: parseInt(transformerEpochs),
                        sequenceLength: parseInt(cnnSequenceLength), // Transformer için sequenceLength
                    };
                    break;
                case 'sarima':
                    modelParams.params = {
                        p: parseInt(sarimaP),
                        d: parseInt(sarimaD),
                        q: parseInt(sarimaQ),
                        P: parseInt(sarimaSP),
                        D: parseInt(sarimaSD),
                        Q: parseInt(sarimaSQ),
                        s: parseInt(sarimaSeasonalPeriod),
                    };
                    break;
                case 'random_forest':
                    modelParams.params = {
                        nEstimators: parseInt(rfEstimators),
                        maxDepth: rfMaxDepth === null ? null : parseInt(rfMaxDepth),
                        minSamplesSplit: parseInt(rfMinSamplesSplit),
                        minSamplesLeaf: parseInt(rfMinSamplesLeaf),
                        sequenceLength: parseInt(cnnSequenceLength), // RF için sequenceLength (lag oluşturma)
                    };
                    break;
                case 'rocket':
                    modelParams.params = {
                        num_kernels: parseInt(rocketNumKernels),
                        window_size: parseInt(rocketWindowSize),
                        stride: parseInt(rocketStride),
                        sequenceLength: parseInt(cnnSequenceLength), // ROCKET için sequenceLength (pencereleme)
                    };
                    break;
                case 'inception_time':
                    modelParams.params = {
                        nb_filters: parseInt(inceptionNbFilters),
                        use_residual: inceptionUseResidual,
                        use_bottleneck: inceptionUseBottleneck,
                        depth: parseInt(inceptionDepth),
                        kernel_size: parseInt(inceptionKernelSize),
                        epochs: parseInt(inceptionEpochs),
                        sequenceLength: parseInt(cnnSequenceLength), // InceptionTime için sequenceLength
                    };
                    break;
                case 'shapelet_transform':
                    modelParams.params = {
                        n_shapelets: parseInt(shapeletNShapelets),
                        shapelet_lengths: shapeletLengths.map(parseFloat),
                        window_size: parseInt(shapeletSequenceLength),
                        stride: Math.max(1, parseInt(shapeletSequenceLength / 10)),
                    };
                    break;
                default:
                    throw new Error('Unknown model type');
            }

            const response = await fetch('http://localhost:5000/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(modelParams),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Model training failed');
            }

            const results = await response.json();
            setResults(results);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleShapeletLengthsChange = (e) => {
        const value = e.target.value.split(',').map(v => v.trim()).filter(v => v !== '');
        setShapeletLengths(value);
    };

    return (
        <div className="container"
             style={{fontFamily: 'Arial, sans-serif', padding: '20px',marginTop:"60px"}}>

            {/* File Upload */}
            <div style={{
                marginBottom: '20px',
                padding: '15px',
                backgroundColor: '#fff',
                borderRadius: '5px',
                border: '1px solid #ddd'
            }}>
                <label htmlFor="file-upload" style={{display: 'block', marginBottom: '10px', fontWeight: 'bold'}}>Upload
                    CSV File:</label>
                <input
                    id="file-upload"
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    style={{padding: '8px', border: '1px solid #ccc', borderRadius: '3px', width: 'calc(100% - 20px)'}}
                />
                <button
                    onClick={handleUpload}
                    disabled={isLoading}
                    style={{
                        marginTop: '10px',
                        padding: '10px 15px',
                        backgroundColor: isLoading ? '#ccc' : '#5cb85c',
                        color: 'white',
                        border: 'none',
                        borderRadius: '3px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontSize: '1em',
                    }}
                >
                    {isLoading ? 'Uploading...' : 'Upload'}
                </button>
            </div>

            {/* Show error */}
            {error && <div style={{color: 'red', marginBottom: '15px', fontWeight: 'bold'}}>Error: {error}</div>}
            {/* Show columns and preview if uploaded */}
            {columns.length > 0 && (
                <div style={{
                    marginBottom: '20px',
                    padding: '15px',
                    backgroundColor: '#fff',
                    borderRadius: '5px',
                    border: '1px solid #ddd'
                }}>
                    <h3 style={{color: '#333', marginBottom: '10px'}}>Select Target Column</h3>
                    <select
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        style={{
                            padding: '8px',
                            border: '1px solid #ccc',
                            borderRadius: '3px',
                            width: 'calc(100% - 20px)'
                        }}
                    >
                        {columns.map((col) => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                </div>
            )}

            {/* Preview Data */}
            {previewData && (
                <div style={{
                    marginBottom: '20px',
                    padding: '15px',
                    backgroundColor: '#fff',
                    borderRadius: '5px',
                    border: '1px solid #ddd',
                    overflowX: 'auto'
                }}>
                    <h3 style={{color: '#333', marginBottom: '10px'}}>Data Preview</h3>
                    <table border="1" style={{width: '100%', borderCollapse: 'collapse'}}>
                        <thead>
                        <tr style={{backgroundColor: '#f0f0f0'}}>
                            {columns.map((col) => (
                                <th key={col}
                                    style={{padding: '8px', border: '1px solid #ccc', textAlign: 'left'}}>{col}</th>
                            ))}
                        </tr>
                        </thead>
                        <tbody>
                        {
                            previewData.map((row, idx) => (
                            <tr key={idx}>
                                {columns.map((col) => (
                                    <td key={col} style={{padding: '8px', border: '1px solid #ccc'}}>{row[col]}</td>
                                ))}
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Model Type Selection */}
            <div style={{
                marginBottom: '20px',
                padding: '15px',
                backgroundColor: '#fff',
                borderRadius: '5px',
                border: '1px solid #ddd'
            }}>
                <h3 style={{color: '#333', marginBottom: '10px'}}>Select Model Type</h3>
                <select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value)}
                    style={{padding: '8px', border: '1px solid #ccc', borderRadius: '3px', width: 'calc(100% - 20px)'}}
                >
                    <option value="cnn">CNN</option>
                    <option value="lstm">LSTM</option>
                    <option value="arima">ARIMA</option>
                    <option value="prophet">Prophet</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="transformer">Transformer</option>
                    <option value="sarima">SARIMA</option>
                    <option value="random_forest">Random Forest</option>
                    <option value="rocket">ROCKET</option>
                    <option value="inception_time">InceptionTime</option>
                    <option value="shapelet_transform">Shapelet Transform</option>
                </select>
            </div>

            {/* Parameters Input Section */}
            <div className="parameters-section" style={{
                marginBottom: '20px',
                padding: '15px',
                backgroundColor: '#fff',
                borderRadius: '5px',
                border: '1px solid #ddd'
            }}>
                {modelType === 'cnn' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>CNN Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Sequence Length:
                            <input type="number" min="1" value={cnnSequenceLength}
                                   onChange={(e) => setCnnSequenceLength(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Num Filters:
                            <input type="number" min="1" value={cnnNumFilters}
                                   onChange={(e) => setCnnNumFilters(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Kernel Size:
                            <input type="number" min="1" value={cnnKernelSize}
                                   onChange={(e) => setCnnKernelSize(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Dense Units:
                            <input type="number" min="1" value={cnnDenseUnits}
                                   onChange={(e) => setCnnDenseUnits(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Epochs:
                            <input type="number" min="1" value={cnnEpochs}
                                   onChange={(e) => setCnnEpochs(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* LSTM Parameters */}
                {modelType === 'lstm' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>LSTM Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            LSTM Units:
                            <input type="number" min="1" value={lstmUnits}
                                   onChange={(e) => setLstmUnits(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Dense Units:
                            <input type="number" min="1" value={lstmDenseUnits}
                                   onChange={(e) => setLstmDenseUnits(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Epochs:
                            <input type="number" min="1" value={lstmEpochs}
                                   onChange={(e) => setLstmEpochs(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* ARIMA Parameters */}
                {modelType === 'arima' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>ARIMA Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            p (AR order):
                            <input type="number" min="0" value={arimaPValue}
                                   onChange={(e) => setArimaPValue(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            d (Difference order):
                            <input type="number" min="0" value={arimaDValue}
                                   onChange={(e) => setArimaDValue(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            q (MA order):
                            <input type="number" min="0" value={arimaQValue}
                                   onChange={(e) => setArimaQValue(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* Prophet Parameters */}
                {modelType === 'prophet' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>Prophet Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Seasonality Mode:
                            <select value={prophetSeasonalityMode}
                                    onChange={(e) => setProphetSeasonalityMode(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '150px'
                            }}>
                                <option value="additive">Additive</option>
                                <option value="multiplicative">Multiplicative</option>
                            </select>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Change Point Prior Scale:
                            <input type="number" step="0.01" min="0" value={prophetChangePointPrior}
                                   onChange={(e) => setProphetChangePointPrior(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Seasonality Prior Scale:
                            <input type="number" step="0.1" min="0" value={prophetSeasonalityPrior}
                                   onChange={(e) => setProphetSeasonalityPrior(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* XGBoost Parameters */}
                {modelType === 'xgboost' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>XGBoost Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Max Depth:
                            <input type="number" min="1" value={xgboostMaxDepth}
                                   onChange={(e) => setXgboostMaxDepth(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Learning Rate:
                            <input type="number" step="0.01" min="0" value={xgboostLearningRate}
                                   onChange={(e) => setXgboostLearningRate(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Estimators:
                            <input type="number" min="1" value={xgboostNEstimators}
                                   onChange={(e) => setXgboostNEstimators(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* Transformer Parameters */}
                {modelType === 'transformer' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>Transformer Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Heads:
                            <input type="number" min="1" value={transformerNumHeads}
                                   onChange={(e) => setTransformerNumHeads(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Encoder Layers:
                            <input type="number" min="1" value={transformerNumEncoderLayers}
                                   onChange={(e) => setTransformerNumEncoderLayers(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Dropout Rate:
                            <input type="number" step="0.01" min="0" max="1" value={transformerDropoutRate}
                                   onChange={(e) => setTransformerDropoutRate(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Dimension of Model:
                            <input type="number" min="1" value={transformerDimModel}
                                   onChange={(e) => setTransformerDimModel(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Epochs:
                            <input type="number" min="1" value={transformerEpochs}
                                   onChange={(e) => setTransformerEpochs(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* SARIMA Parameters */}
                {modelType === 'sarima' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>SARIMA Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            p (AR order):
                            <input type="number" min="0" value={sarimaP} onChange={(e) => setSarimaP(e.target.value)}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '80px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            d (Difference order):
                            <input type="number" min="0" value={sarimaD} onChange={(e) => setSarimaD(e.target.value)}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '80px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            q (MA order):
                            <input type="number" min="0" value={sarimaQ} onChange={(e) => setSarimaQ(e.target.value)}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '80px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            P (Seasonal AR order):
                            <input type="number" min="0" value={sarimaSP} onChange={(e) => setSarimaSP(e.target.value)}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '80px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            D (Seasonal difference order):
                            <input type="number" min="0" value={sarimaSD} onChange={(e) => setSarimaSD(e.target.value)}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '80px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Q (Seasonal MA order):
                            <input type="number" min="0" value={sarimaSQ} onChange={(e) => setSarimaSQ(e.target.value)}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '80px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Seasonal Period:
                            <input type="number" min="1" value={sarimaSeasonalPeriod}
                                   onChange={(e) => setSarimaSeasonalPeriod(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* Random Forest Parameters */}
                {modelType === 'random_forest' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>Random Forest Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Estimators:
                            <input type="number" min="1" value={rfEstimators}
                                   onChange={(e) => setRfEstimators(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Max Depth:
                            <input
                                type="number"
                                min="1"
                                value={rfMaxDepth === null ? '' : rfMaxDepth}
                                onChange={(e) => setRfMaxDepth(e.target.value === '' ? null : parseInt(e.target.value))}
                                placeholder="Leave empty for None"
                                style={{
                                    marginLeft: '10px',
                                    padding: '6px',
                                    border: '1px solid #ccc',
                                    borderRadius: '3px',
                                    width: '150px'
                                }}
                            />
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Min Samples Split:
                            <input type="number" min="2" value={rfMinSamplesSplit}
                                   onChange={(e) => setRfMinSamplesSplit(parseInt(e.target.value))} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Min Samples Leaf:
                            <input type="number" min="1" value={rfMinSamplesLeaf}
                                   onChange={(e) => setRfMinSamplesLeaf(parseInt(e.target.value))} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* ROCKET Parameters */}
                {modelType === 'rocket' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>ROCKET Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Kernels:
                            <input type="number" min="1" value={rocketNumKernels}
                                   onChange={(e) => setRocketNumKernels(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Window Size:
                            <input type="number" min="1" value={rocketWindowSize}
                                   onChange={(e) => setRocketWindowSize(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Stride:
                            <input type="number" min="1" value={rocketStride}
                                   onChange={(e) => setRocketStride(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* InceptionTime Parameters */}
                {modelType === 'inception_time' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>InceptionTime Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Filters:
                            <input type="number" min="1" value={inceptionNbFilters}
                                   onChange={(e) => setInceptionNbFilters(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Use Residual:
                            <select value={inceptionUseResidual}
                                    onChange={(e) => setInceptionUseResidual(e.target.value === 'true')} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '100px'
                            }}>
                                <option value="true">True</option>
                                <option value="false">False</option>
                            </select>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Use Bottleneck:
                            <select value={inceptionUseBottleneck}
                                    onChange={(e) => setInceptionUseBottleneck(e.target.value === 'true')} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '100px'
                            }}>
                                <option value="true">True</option>
                                <option value="false">False</option>
                            </select>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Depth:
                            <input type="number" min="1" value={inceptionDepth}
                                   onChange={(e) => setInceptionDepth(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Kernel Size:
                            <input type="number" min="1" value={inceptionKernelSize}
                                   onChange={(e) => setInceptionKernelSize(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Epochs:
                            <input type="number" min="1" value={inceptionEpochs}
                                   onChange={(e) => setInceptionEpochs(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}

                {/* Shapelet Transform Parameters */}
                {modelType === 'shapelet_transform' && (
                    <>
                        <h3 style={{color: '#333', marginBottom: '10px'}}>Shapelet Transform Parameters</h3>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Number of Shapelets:
                            <input type="number" min="1" value={shapeletNShapelets}
                                   onChange={(e) => setShapeletNShapelets(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Shapelet Lengths (comma-separated decimals):
                            <input type="text" value={shapeletLengths.join(',')} onChange={handleShapeletLengthsChange}
                                   style={{
                                       marginLeft: '10px',
                                       padding: '6px',
                                       border: '1px solid #ccc',
                                       borderRadius: '3px',
                                       width: '200px'
                                   }}/>
                        </label>
                        <label style={{display: 'block', marginBottom: '8px'}}>
                            Sequence Length (for windowing):
                            <input type="number" min="1" value={shapeletSequenceLength}
                                   onChange={(e) => setShapeletSequenceLength(e.target.value)} style={{
                                marginLeft: '10px',
                                padding: '6px',
                                border: '1px solid #ccc',
                                borderRadius: '3px',
                                width: '80px'
                            }}/>
                        </label>
                    </>
                )}
            </div>

            {/* Train Button */}
            <div style={{
                marginBottom: '20px',
                padding: '15px',
                backgroundColor: '#fff',
                borderRadius: '5px',
                border: '1px solid #ddd',
                textAlign: 'center'
            }}>
                <button
                    onClick={handleModelTrain}
                    disabled={isLoading}
                    style={{
                        padding: '12px 20px',
                        backgroundColor: isLoading ? '#ccc' : '#7510a8',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontSize: '1.1em',
                    }}
                >
                    {isLoading ? 'Training...' : 'Train Model'}
                </button>
            </div>

            {/* Results Section */}
            {results && (
                <div style={{
                    padding: '20px',
                    backgroundColor: '#e9ecef',
                    borderRadius: '5px',
                    border: '1px solid #ddd'
                }}>
                    <h3 style={{color: '#333', marginBottom: '15px', textAlign: 'center'}}>Training Results</h3>
                    {results.model && <p><strong>Model:</strong> {results.model}</p>}
                    {results.mse && <p><strong>MSE:</strong> {results.mse.toFixed(4)}</p>}
                    {results.rmse && <p><strong>RMSE:</strong> {results.rmse.toFixed(4)}</p>}
                    {results.mape &&
                        <p><strong>MAPE:</strong> {results.mape ? results.mape.toFixed(2) + '%' : 'N/A'}</p>}
                    {results.aic && <p><strong>AIC:</strong> {results.aic ? results.aic.toFixed(2) : 'N/A'}</p>}
                    {results.accuracy && <p>
                        <strong>Accuracy:</strong> {results.accuracy ? (results.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                    </p>}
                    {results.trainingTime &&
                        <p><strong>Training Time:</strong> {results.trainingTime.toFixed(2)} seconds</p>}

                    {/* Model Tahmin Grafiği */}
                    {results.plot && (
                        <div style={{
                            marginTop: '15px',
                            border: '1px solid #ccc',
                            borderRadius: '3px',
                            overflow: 'hidden'
                        }}>
                            <h4 style={{
                                backgroundColor: '#f8f9fa',
                                padding: '10px',
                                borderBottom: '1px solid #eee'
                            }}>Model Prediction Plot</h4>
                            <img
                                src={`data:image/png;base64,${results.plot}`}
                                alt="Model Plot"
                                style={{maxWidth: '100%', height: 'auto', display: 'block'}}
                            />
                        </div>
                    )}

                    {/* Feature Importance Grafiği */}
                    {results.featureImportance && (
                        <div style={{
                            marginTop: '15px',
                            border: '1px solid #ccc',
                            borderRadius: '3px',
                            overflow: 'hidden'
                        }}>
                            <h4 style={{
                                backgroundColor: '#f8f9fa',
                                padding: '10px',
                                borderBottom: '1px solid #eee'
                            }}>Feature Importance</h4>
                            <img
                                src={`data:image/png;base64,${results.featureImportance}`}
                                alt="Feature Importance Plot"
                                style={{maxWidth: '100%', height: 'auto', display: 'block'}}
                            />
                        </div>
                    )}

                    {/* Training History */}
                    {results.history && (
                        <div style={{
                            marginTop: '15px',
                            border: '1px solid #ccc',
                            borderRadius: '3px',
                            padding: '10px',
                            backgroundColor: '#f8f9fa'
                        }}>
                            <h4 style={{
                                marginBottom: '10px',
                                borderBottom: '1px solid #eee',
                                paddingBottom: '5px'
                            }}>Training History</h4>
                            {results.history.loss && (
                                <div>
                                    <strong>Loss:</strong>
                                    <pre style={{
                                        backgroundColor: '#fff',
                                        padding: '10px',
                                        borderRadius: '3px',
                                        overflowX: 'auto'
                                    }}>{JSON.stringify(results.history.loss, null, 2)}</pre>
                                </div>
                            )}
                            {results.history.val_loss && (
                                <div>
                                    <strong>Validation Loss:</strong>
                                    <pre style={{
                                        backgroundColor: '#fff',
                                        padding: '10px',
                                        borderRadius: '3px',
                                        overflowX: 'auto'
                                    }}>{JSON.stringify(results.history.val_loss, null, 2)}</pre>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Diğer Sonuçlar (JSON formatında) */}
                    {/* <pre>{JSON.stringify(results, null, 2)}</pre> */}
                </div>
            )}
            <Particle/>
        </div>
    );
}

export default Projects;
