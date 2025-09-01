import React, { createContext, useContext, useState } from 'react';

const GlobalContext = createContext();

export const GlobalProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [modifiedFile, setModifiedFile] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);

  const [isModifiedByAI, setIsModifiedByAI] = useState(false);

  return (
    <GlobalContext.Provider value={{
      messages,
      setMessages,
      originalFile,
      setOriginalFile,
      modifiedFile,
      setModifiedFile,
      pdfFile,
      setPdfFile,
      isModifiedByAI,
      setIsModifiedByAI
    }}>
      {children}
    </GlobalContext.Provider>
  );
};

export const useGlobal = () => useContext(GlobalContext);