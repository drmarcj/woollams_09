#!/bin/tcsh

foreach f (1b 2 3 4 5 6 7 8 10)


eval_speaking_ministep -weight s${f}/3250000_weights -pat testing.gen-mini.pat | sort | join -1 1 -2 1 - testing-items-sorted > s$f.3.25.gen-mini.out

eval_speaking_ministep -weight s${f}/3250000_weights -pat testing.speaking-mini.pat | sort | join -1 1 -2 1 - testing-items-past-sorted > s$f.3.25.sp-mini.out

end
