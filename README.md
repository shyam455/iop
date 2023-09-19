This is a Flask Web App.
In this application user can enter different parameters of an extra High Votage Transmission Line.
![Fig1](https://github.com/shyam455/iop/assets/135046173/df18084f-b9ef-48ed-a5a5-52f8cdff1bbe)
Input Parameters:-
             1. See in Fig1 suppose lines are like that and enter (x's, Y's) of all lines.
             2. Enter the (x,y) coordinate on the x-y plane of a specific point in the space where we want to measure ouput parameters.
             3. Enter the range as far as we want to measure the output parameters.
             4. Line voltage(in rms(kV)).
             5. Total power suppling to the load(in MW)
             6. Number of Sub-condutors in every bundle of every conductor
             7. Diameter of Sub-conductor in bundle(in cm)
             8. Bundle spacing(in cm)

Output Parameters:-
             1. Maximum voltage gradient for outer lines(line1 and line3)
             2. Maximum voltage gradient for middle line(line2)
             3. Audible Noise due to line-1, line-2 and line-3
             4. Radio Interference due to line-1, line-2 and line-3
             5. Total Radio Interference due to all 3 lines
             6. Horizontal, Vertical components and Total magnitude of Electric and Magnetic Field
               Note- (2, 3, 4, 5, 6 these all ouput parameter are at the specific point)
             7. Graph of varying ouput values of Audible-Noise, Radio-Interference, Electric and Magnetic Field.


Tech Stack:-
           1. HTML(html-form) - Taking input parameters
           2. CSS
           3. Flask(Backend) - To calculate output parameters and data for graphs
                  (i) Numpy
                  (ii) Jinja Templating - To display output parameters and graphs on same input form display(html page)
           4. Chart.js - To display graphs
