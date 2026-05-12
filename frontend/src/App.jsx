import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [image, setImage] = useState(null)
  const [fileObject, setFileObject] = useState(null)
  const [status, setStatus] = useState('idle') 
  
  const [doctorId, setDoctorId] = useState("DR-JOHN-DOE") // Concept context
  
  // Local history state array
  const [historyItems, setHistoryItems] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)
  const [currentViewData, setCurrentViewData] = useState(null)

  // Doctor's adjusted truths
  const [doctorFeedback, setDoctorFeedback] = useState({})
  
  const [isDragging, setIsDragging] = useState(false)
  const [apiPort, setApiPort] = useState("8000") // Dynamic Node Routing
  
  const fileInputRef = useRef(null)

  // Load History from Browser Storage bounded by current API Port
  useEffect(() => {
     const historyKey = `dacnet_history_${apiPort}`
     const saved = localStorage.getItem(historyKey)
     if (saved) {
         setHistoryItems(JSON.parse(saved))
     } else {
         setHistoryItems([])
     }
  }, [apiPort])
  
  // Persist History updates uniquely to the active Node
  const updateHistoryState = (newItems) => {
      setHistoryItems(newItems)
      localStorage.setItem(`dacnet_history_${apiPort}`, JSON.stringify(newItems))
  }

  const processFile = (file) => {
    if (file) {
      const objUrl = URL.createObjectURL(file)
      setImage(objUrl)
      setFileObject(file)
      setStatus('idle')
      setCurrentViewData(null)
      setSelectedHistoryId(null)
      setDoctorFeedback({})
    }
  }

  const handleImageUpload = (e) => {
    processFile(e.target.files[0])
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleAnalyze = async () => {
    if (!fileObject) return;
    setStatus('processing')
    
    const formData = new FormData();
    formData.append("file", fileObject);
    
    try {
      const response = await fetch(`http://localhost:${apiPort}/analyze`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      
      if (data.status === "success") {
         
         const newHistoryItem = {
             id: data.job_id,
             filename: data.filename,
             timestamp: new Date().toLocaleString(),
             pathologies: data.pathologies,
             heatmap_base64: data.heatmap_base64,
             ssh_hash: data.ssh_hash,
             cid: data.cid,
             tx_id: data.tx_id,
             train_status: 'pending', // pending | trained
             preview_img_url: `http://localhost:${apiPort}/image/${data.job_id}`
         };

         // Prepend to history array
         const updatedList = [newHistoryItem, ...historyItems];
         updateHistoryState(updatedList);
         
         // Set View State to this new item instantly
         loadHistoryItemToView(newHistoryItem);

      } else {
         console.error(data.message);
         setStatus('idle')
      }
    } catch (error) {
      console.error("API Connection Failed", error);
      setStatus('idle')
    }
  }

  const loadHistoryItemToView = (item) => {
      setSelectedHistoryId(item.id)
      setCurrentViewData(item)
      setImage(item.preview_img_url)
      
      let initialFeedback = {};
      item.pathologies.forEach(p => {
          initialFeedback[p.name] = p.score;
      });
      
      // If history already has saved feedback limits, use those. 
      // Else use AI baseline limits.
      setDoctorFeedback(item.savedFeedback || initialFeedback);
      
      if (item.train_status === 'trained') {
          setStatus('learned')
      } else {
          setStatus('complete')
      }
  }

  const clearHistory = () => {
      setHistoryItems([])
      localStorage.removeItem(`dacnet_history_${apiPort}`)
      setImage(null)
      setFileObject(null)
      setCurrentViewData(null)
      setSelectedHistoryId(null)
      setDoctorFeedback({})
      setStatus('idle')
  }

  const handleTrain = async () => {
      if (!currentViewData) return;
      setStatus('training')
      
      try {
          const payload = {
              job_id: currentViewData.id,
              confirmed_pathologies: doctorFeedback
          }
          
          const response = await fetch(`http://localhost:${apiPort}/train`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
          });
          const data = await response.json();
          
          if (data.status === "success") {
              const updatedHistory = historyItems.map(item => {
                  if (item.id === currentViewData.id) {
                      return { 
                          ...item, 
                          train_status: 'trained', 
                          savedFeedback: doctorFeedback,
                          final_loss: data.loss
                      }
                  }
                  return item;
              })
              updateHistoryState(updatedHistory)
              
              // Refresh view
              loadHistoryItemToView(updatedHistory.find(i => i.id === currentViewData.id))
          } else {
              console.error(data.message);
              setStatus('complete'); 
          }
      } catch (error) {
          console.error("Training API Failed", error);
          setStatus('complete');
      }
  }

  return (
    <>
      <div className="header-container animate-slide-up" style={{position: "relative"}}>
        
        <div style={{position: "absolute", right: "0", top: "0", display: "flex", gap: "0.5rem", alignItems: "center", background: "rgba(0,0,0,0.2)", padding: "0.5rem 1rem", borderRadius: "8px", border: "1px solid var(--glass-border)"}}>
            <span style={{fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: "bold"}}>🔗 TARGET API PORT:</span>
            <input 
              type="text" 
              value={apiPort} 
              onChange={(e) => setApiPort(e.target.value)} 
              style={{background: "var(--bg-dark)", color: "var(--accent-primary)", border: "1px solid var(--glass-border)", padding: "0.2rem", width: "60px", borderRadius: "4px", textAlign: "center", fontWeight: "bold"}} 
            />
        </div>

        <h1 className="header-title glow-text">DacNet Clinical Dashboard</h1>
        <p className="header-subtitle">Upload & Bank X-Rays. Return later to confirm diagnoses.</p>
      </div>

      <div className="app-container">
        
        {/* Left Column: History Bank */}
        <div className="history-container glass-panel animate-slide-up" style={{ padding: '1rem', animationDelay: '0s' }}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: "1rem", borderBottom: "1px solid var(--glass-border)", paddingBottom: "0.5rem"}}>
                <h3 style={{fontSize: "1.1rem", margin: 0}}>
                    Patient History Vault
                </h3>
                {historyItems.length > 0 && (
                    <button 
                        onClick={clearHistory}
                        style={{background: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444', color: '#ef4444', padding: '0.2rem 0.5rem', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 'bold'}}
                    >
                        Clear Vault
                    </button>
                )}
            </div>
            
            {historyItems.length === 0 ? (
                <div style={{color: "var(--text-muted)", fontSize: "0.9rem", textAlign: "center", marginTop: "2rem"}}>
                   No open cases.<br/>Upload an X-Ray.
                </div>
            ) : null}

            {historyItems.map((item) => (
                <div 
                    key={item.id} 
                    className={`history-card ${selectedHistoryId === item.id ? 'active' : ''} ${item.train_status === 'trained' ? 'resolved' : ''}`}
                    onClick={() => loadHistoryItemToView(item)}
                >
                    <div className="history-title">{item.filename}</div>
                    <div style={{color: "var(--text-muted)", fontSize: "0.75rem", marginBottom: "0.5rem"}}>
                        {item.timestamp}
                    </div>
                    <div className={`history-status ${item.train_status === 'pending' ? 'status-pending' : 'status-trained'}`}>
                        {item.train_status === 'pending' ? '⏳ Pending Physicals' : '✅ Learned & Closed'}
                    </div>
                </div>
            ))}
        </div>

        {/* Center Column: Upload & Image Viewer */}
        <div className="glass-panel animate-slide-up" style={{ animationDelay: '0.1s', display: 'flex', flexDirection: 'column'}}>
          <div 
            className={`upload-zone ${isDragging ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {image ? (
              <div style={{display: 'flex', gap: '1rem', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%', position: 'relative', zIndex: 10}}>
                <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, height: '100%', padding: '1rem'}}>
                  <span style={{fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem'}}>Original X-Ray</span>
                  <img src={image} alt="X-Ray Preview" className="preview-image-flex" style={{maxHeight: '300px'}} />
                </div>
                {(status === 'complete' || status === 'learned') && currentViewData?.heatmap_base64 && (
                   <div className="animate-slide-up" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1, height: '100%', padding: '1rem'}}>
                     <span style={{fontSize: '0.8rem', color: 'var(--accent-primary)', marginBottom: '0.5rem', fontWeight: 'bold'}}>✨ AI Saliency Map</span>
                     <img src={`data:image/png;base64,${currentViewData.heatmap_base64}`} alt="AI Heatmap" className="preview-image-flex" style={{maxHeight: '300px', border: '2px solid var(--accent-primary)', boxShadow: '0 0 15px rgba(59, 130, 246, 0.3)'}} />
                   </div>
                )}
              </div>
            ) : (
              <>
                <div className="upload-icon">☢️</div>
                <div className="upload-text">Upload Patient X-Ray</div>
                <div className="upload-subtext">Click to ingest new diagnostic</div>
              </>
            )}
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onChange={handleImageUpload}
              id="file-input-trigger"
            />
          </div>

          <button 
            className={`btn-primary ${status === 'processing' || status === 'training' ? 'processing' : ''}`}
            disabled={(!image && status !== 'complete' && status !== 'learned') || status === 'processing' || status === 'training'}
            onClick={
                status === 'idle' ? handleAnalyze : 
                () => document.getElementById("file-input-trigger").click()
            }
          >
            {status === 'idle' && 'Push to Secure Processing Network'}
            {status === 'processing' && 'Processing Pipeline...'}
            {(status === 'complete' || status === 'learned') && 'Ingest Another Patient Scan'}
            {status === 'training' && 'Executing Local Backpropagation...'}
          </button>
        </div>

        {/* Right Column: Active Feedback Loop */}
        <div className="glass-panel animate-slide-up" style={{ animationDelay: '0.2s', display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: "auto", maxHeight: "650px" }}>
          
          {/* Active Learning Form Container */}
          <div className={`result-card ${status === 'complete' || status === 'learned' ? 'success' : ''}`}>
             <div className="result-title">🎯 Doctor Diagnostic Truth (Delayed Update)</div>
             {status === 'processing' && <div className="result-value">Inferencing...</div>}
             {status === 'idle' && <div className="result-value" style={{color: 'var(--text-muted)'}}>Select a case from history to verify.</div>}
             
             {(status === 'complete' || status === 'learned') && currentViewData && (
               <div className="pathologies-list">
                 <p style={{fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1rem"}}>
                     Run your physical real-world tests. Return here to formally lock the exact pathology distributions.
                 </p>
                 
                 {status === 'complete' && (
                     <button onClick={handleTrain} className="btn-primary" style={{marginTop: "0", marginBottom: "1.5rem", background: "var(--accent-success)", padding: "0.8rem", fontSize: "1rem" }}>
                         Confirm Final Truth & Update AI
                     </button>
                 )}
                 
                 {status === 'learned' && (
                     <div style={{marginTop: "0", marginBottom: "1.5rem", color: "var(--accent-success)", fontWeight: "bold", background: "rgba(16, 185, 129, 0.1)", padding: "1rem", borderRadius: "12px", border: "1px solid var(--accent-success)"}}>
                         ✅ Neural Weights Adjusted.<br/>
                         Job Vault Closed natively.<br/>
                         Loss Gradient: {currentViewData.final_loss?.toFixed(4)}
                     </div>
                 )}

                 <div style={{display: "flex", flexDirection: "column", gap: "1rem", maxHeight: "400px", overflowY: "auto", paddingRight: "0.5rem"}}>
                   {currentViewData.pathologies.map((path, idx) => (
                     <div key={idx} className="pathology-row">
                       <div className="path-header">
                         <span>{path.name} (AI Base: {(path.score * 100).toFixed(1)}%)</span>
                         <span>Final Limit: {(doctorFeedback[path.name] * 100).toFixed(1)}%</span>
                       </div>
                       <input 
                          type="range" 
                          min="0" max="1" step="0.01" 
                          value={doctorFeedback[path.name] ?? path.score}
                          onChange={(e) => {
                              setDoctorFeedback(prev => ({
                                  ...prev, [path.name]: parseFloat(e.target.value)
                              }))
                          }}
                          style={{width: "100%", accentColor: "var(--accent-primary)", opacity: status === 'learned' ? 0.5 : 1}}
                          disabled={status === 'learned'}
                       />
                     </div>
                   ))}
                 </div>
               </div>
             )}
          </div>

          <div className={`result-card ${status === 'complete' || status === 'learned' ? 'success' : ''}`}>
             <div className="result-title">🔐 Cryptographic Identity</div>
             <div className="result-value" style={{fontSize: "0.9rem"}}>
               {status === 'processing' && <span>Generating Hash...</span>}
               {(status === 'complete' || status === 'learned') && currentViewData?.ssh_hash}
             </div>
          </div>

          <div className={`result-card ${status === 'complete' || status === 'learned' ? 'success' : ''}`}>
             <div className="result-title">🔗 IPFS & Ledger Verification</div>
             <div className="result-value" style={{fontSize: "0.9rem"}}>
               {status === 'processing' && <span>Awaiting Architecture...</span>}
               {(status === 'complete' || status === 'learned') && (
                   <div style={{display: "flex", flexDirection: "column", gap: "0.5rem"}}>
                       <span><strong>CID Pinned:</strong> {currentViewData?.cid}</span>
                       <span><strong>Logic TX ID:</strong> {currentViewData?.tx_id}</span>
                   </div>
               )}
             </div>
          </div>

        </div>
      </div>
    </>
  )
}

export default App
