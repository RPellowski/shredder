# Progress log

## Week 8 Retrospective

### The good

* Orientation angles are determined on 79% of pieces
* Have an algorithm that sets polarity on over half those pieces

### The bad

* Not all pieces get the correct polarity based on black ink
* Not all pieces have black ink
* Only 47% of total pieces have angle and polarity automatically determined
* Less development time spent on project
* No velocity

### Both good and bad

* Noticed that concave/convex top and bottom edges are good indication of
  polarity, need to pivot with a story for that

## Week 7 Retrospective

### The good

* Refocused efforts on customer value of stories and size of stories
* Algorithm automatically solves orientation for 78% of the pieces

### The bad

* Passed the historic first solver milestone without even solving the first puzzle
* Wasted a lot of time on a story involving interactive angle selection
* No velocity for weeks

### Both good and bad

* Learned Python matplotlib widgets, including bugs and quirks

## Week 3 Retrospective

### The good

* Have captured a good amount of domain knowledge in code

### The bad

* No completed story points for this iteration
* Started investigation into Python imaging software for the next stories- taking away from current work

## Week 2 Retrospective

### The good

* Completed automatic filesystem activity logging and have it assigning basic
  categories- development, documentation, other

### The bad

* No reports and no meaningful progress on the imaging side

## Week 1 Retrospective

### The good

* Completed a sprint with a good backlog and some coding done
* Have a strategy
    * what is the ideal (whole sheet of notepad paper with writing)
    * what is the the cost function to get to the ideal
    * how can we use the cost function to drive the solution based on iterating measurements and algorithms

### The bad

* Didn't complete as many stories or spend as much time as I would have liked
* Definition of done does not include CI/CD
* Worried about shortcomings of current approach- how to keep it polynomial, especially for larger puzzles
* One week out of 7 down with no puzzles solved and no reports

### Both good and bad

* Value is long term (logging and reporting only) instead of immediate (doing something with data)
* Committed to tools which would normally deserve some research into alternatives
    * Sprint tool ([https://www.pivotaltracker.com](PivotalTracker)) is different than what I am used to
    * Language (Python, because it supports PIL for imaging and NumPy/SciPy)
    * GitHub, which does not have built-in CI/CD

