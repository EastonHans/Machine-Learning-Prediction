from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("predict.html", errors=None)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            humidity = request.form['Humidity']
            reading = request.form['Reading']
            temperature = request.form['Temp.']
            wind_direction = request.form['Wind dir.']

            # Validate input values
            errors = {}
            if not humidity:
                errors['Humidity'] = "Humidity cannot be blank, please enter a value."
            elif not humidity.isdigit():
                errors['Humidity'] = "Humidity must be a number."

            elif not (0 <= float(humidity) <= 100):
                errors['Humidity'] = "Your humidity value is out of range."

            if not reading:
                errors['Reading'] = "Barometer reading cannot be blank, please enter a value."
            elif not reading.replace('.', '', 1).isdigit():
                errors['Reading'] = "Barometer reading must be a number."
            elif not (0 <= float(reading) <= 1500):
                errors['Reading'] = "Your barometer reading is out of range. Enter a value between 0 and 1500"

            if not temperature:
                errors['Temp.'] = "Temperature value cannot be blank, please enter a value."
            elif not temperature.replace('.', '', 1).isdigit():
                errors['Temp.'] = "Temperature must be a number."
            elif not (0 <= float(temperature) <= 50):
                errors['Temp.'] = "Your temperature value is out of range."

            if not wind_direction:
                errors['Wind dir.'] = "Wind direction cannot be blank, please enter a value."
            elif not wind_direction.isdigit():
                errors['Wind dir.'] = "Wind direction must be a number."
            elif not (0 <= float(wind_direction) <= 360):
                errors['Wind dir.'] = "Wind direction must be between 0 and 360 degrees."

            if errors:
                return render_template('predict.html', errors=errors)

            # Perform prediction
            input_features = [float(humidity), float(reading), float(temperature), float(wind_direction)]
            final = np.array(input_features).reshape(1, -1)
            prediction = model.predict(final)[0]
            rounded_prediction = round(prediction, 1)

            return render_template('predict.html', pred='The expected wind speed is: {:.1f} m/s'.format(prediction))

        except ValueError as e:
            error_message = str(e)
            return render_template('predict.html', errors={'Other': error_message})

    return render_template('predict.html', errors=None)


if __name__ == '__main__':
    app.run(debug=True)
