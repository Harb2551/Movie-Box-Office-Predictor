import React, { useState } from 'react';
import axios from 'axios';

const options = {
  country: ['USA', 'India', 'Japan', 'China'],
  cr: ['G', 'PG', 'PG-13', 'R', 'NC-17'],
  language: ['English', 'Hindi', 'Spanish', 'French', 'German'],
  genres: ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'],
};

const Form = () => {
  const [apidata, setapidata] = useState({})
  const [formData, setFormData] = useState({
    actor_1_name: '',
    actor_2_name: '',
    actor_3_name: '',
    director_name: '',
    country: '',
    cr: '',
    language: '',
    actor_1_likes: '',
    actor_2_facebook_likes: '',
    actor_3_facebook_likes: '',
    director_facebook_likes: '',
    cast_total_facebook_likes: '',
    budget: '',
    gross: '',
    genres: '',
    imdb_score: '',
  });

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData({
      ...formData,
      [name]: value,
    });


  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    console.log(`Movie selected: ${formData}`);
    // try {
    //   await axios.post('/submit-form', formData);
    //   // handle success
    // } catch (error) {
    //   // handle error
    // }





    var myParams = {
      data: formData
    }


    axios.post('/api/query', myParams)
      .then(function (response) {
        console.log(response);
        setapidata(response.data);
        //Perform action based on response
      })
      .catch(function (error) {
        console.log(error);
        //Perform action based on error
      });


  };
  return (
    <>
      <form onSubmit={handleSubmit}>
        <label>
          Actor 1 Name:
          <select name="actor_1_name" value={formData.actor_1_name} onChange={handleChange}>
            <option value="">Select an option</option>
            <option value="Tyler Williams">Tyler Williams</option>
            <option value="Marion Cotillard">Marion Cotillard</option>
            <option value="Jason Segel">Jason Segel</option>
            <option value="Nick Robinson">Nick Robinson</option>
          </select>
        </label>

        <label>
          Actor 2 Name:
          <select name="actor_2_name" value={formData.actor_2_name} onChange={handleChange}>
            <option value="">Select an option</option>
            <option value="Tessa Thompson">Tessa Thompson</option>
            <option value="Joaquin Phoenix">Joaquin Phoenix</option>
            <option value="Jesse Eisenberg">Jesse Eisenberg</option>
            <option value="Gabriel Basso">Gabriel Basso</option>
          </select>
        </label>

        <label>
          Actor 3 Name:
          <select name="actor_3_name" value={formData.actor_3_name} onChange={handleChange}>
            <option value="">Select an option</option>
            <option value="Kyle Gallner">Kyle Gallner</option>
            <option value="Jeremy Renner">Jeremy Renner</option>
            <option value="Anna Chlumsky">Anna Chlumsky</option>
            <option value="Moises Arias">Moises Arias</option>
          </select>
        </label>

        <label>
          Director Name:
          <select name="director_name" value={formData.director_name} onChange={handleChange}>
            <option value="">Select an option</option>
            <option value="Justin Simien">Justin Simien</option>
            <option value="James Gray">James Gray</option>
            <option value="James Ponsoldt">James Ponsoldt</option>
            <option value="Jordan Vogt-Roberts">Jordan Vogt-Roberts</option>
          </select>
        </label>

        <label>
          Country:
          <select name="country" value={formData.country} onChange={handleChange}>
            <option value="">Select a country</option>
            {options.country.map((country) => (
              <option key={country} value={country}>
                {country}
              </option>
            ))}
          </select></label>
        <label>
          contentRating:
          <select name="cr" value={formData.cr} onChange={handleChange}>
            <option value="">cr</option>
            {options.cr.map((cr) => (
              <option key={cr} value={cr}>
                {cr}
              </option>
            ))}
          </select></label>
        <label>
          Language:
          <select name="language" value={formData.language} onChange={handleChange}>
            <option value="">languagey</option>
            {options.language.map((language) => (
              <option key={language} value={language}>
                {language}
              </option>
            ))}
          </select></label>
        <label>
          Budget:
          <input type="number" name="budget" id="budget" onChange={handleChange} />
        </label>
        <label>
          Actor1 Facebook Likes
          <input type="number" name="actor_1_likes" id="actor_1_likes" onChange={handleChange} />
        </label>
        <label>
          Actor2 Facebook Likes:
          <input type="number" name="actor_2_facebook_likes" id="actor_2_facebook_likes" onChange={handleChange} />
        </label>
        <label>
          Actor3 Facebook Likes:
          <input type="number" name="actor_3_facebook_likes" id="actor_3_facebook_likes" onChange={handleChange} />
        </label>
        <label>
          Director Facebook Likes:
          <input type="number" name="director_facebook_likes" id="director_facebook_likes" onChange={handleChange} />
        </label>
        <label>
          Cast Total Facebook Likes:
          <input type="number" name="cast_total_facebook_likes" id="cast_total_facebook_likes" onChange={handleChange} />
        </label>
        <label>
          Gross:
          <input type="number" name="gross" id="gross" onChange={handleChange} />
        </label>
        <label>
          Genre:
          <input type="text" name="genres" id="genres" onChange={handleChange} />
        </label>
        <label>
          IMDB Score:
          <input type="number" step="0.01" name="imdb_score" id="imdb_score" onChange={handleChange} />
        </label>
        <label>
          Aspect Ratio:
          <input type="number" step="0.01" name="imdb_scoree" id="imdb_scoree" onChange={handleChange} />
        </label>
        <label>
          Duration:
          <input type="number" name="imdb_scoreee" id="imdb_scoreeeee" onChange={handleChange} />
        </label>
        <label>
          NumCriticForReviews:
          <input type="numberrr" name="imdb_scorfe" id="imdb_scorewe" onChange={handleChange} />
        </label>
        <label>
          Title-Year:
          <input type="numbeprrr" name="imdb_scoorfe" id="imdb_scporewe" onChange={handleChange} />
        </label>
        <label className="butn-hindi">
          <button type="submit">Submit</button></label>
      </form>


      <div>
        <h2>API Response Data:</h2>
        <ul>
          {Object.keys(apidata).map(key => (
            <li key={key}> {apidata[key]}</li>
          ))}
        </ul>
      </div>
    </>
  )
}
export default Form;