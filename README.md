# Sepsis Metics

This repo contains the work I did as a Data Science Intern with the [Penn Medicine
Predictive Healthcare](http://predictivehealthcare.pennmedicine.org/) team. Its purpose is to showcase my work and not for practical use (although you are welcome to use its contents as you please). The goal was to evaluate the Sepsis Prediction Model over time. However, as detailed in the Patient Selection primer, careful thought was needed to calculate the metrics accurately.

In a nutshell, the task of counting sepsis patients turned out to require some careful thought. During my first attempt, I thought I simply needed to count patients within a desired time interval. It turned out that even this was not straightforward. Since positive sepsis alerts were only counted once for each visit, this naive first attempt would count a patient who received a "positive" alert in one time interval as
"negative" in the next time interval. This would cause increases in the false negative rate, which then decreases the negative predictive value (NPV) over time. After looking at the pros and cons of different selection methods, we decided that I would create a program that would select patients based on when they were discharged. This method had no danger of double counting or excluding patients but required querying databases twice to determine if a patient was discharged.

By implementing the necessary methodological changes mentioned in the Patient Selection primer and making the program more fault-tolerant, I created a project that accurately measured the performance of one of Penn Medicine's life-saving, automated, early-warning systems.

Special thanks goes to [Osama Ahmed](https://www.linkedin.com/in/osamamahmed/) for helping with the patient selection primer, [Michael Becker](https://github.com/mdbecker) for his mentorship, and [Corey Chivers](https://github.com/cjbayesian) for implementing some of the functionality.
