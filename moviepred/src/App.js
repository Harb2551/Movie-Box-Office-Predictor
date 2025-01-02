import logo from './logo.svg';
import './App.css';
import React from 'react';
import MovieForm from './MovieForm';
import Navbar from './NavBar';
// import React from 'react';
function App() {
  return (
    <div className="App">
      <Navbar/>
     <MovieForm/>
    </div>
  );
}

export default App;