import unittest
from fame_workflow import *


class TimelineTestMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_changes(self):
        setup_time = dt.datetime.now()
        setup_value = 0
        setup_rate = 0
        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        _value = timeline.get_value_at(setup_time+dt.timedelta(seconds=60), print_debug=False)
        self.assertEqual(_value, setup_value)

    def test_with_rate(self):
        setup_time = dt.datetime.now()
        setup_value = 0
        setup_rate = -1
        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        setup_dt = 50
        _value = timeline.get_value_at(setup_time+dt.timedelta(seconds=setup_dt), print_debug=False)
        self.assertEqual(_value, setup_value+setup_dt*setup_rate)

    def test_with_reassignment(self):
        setup_time = dt.datetime.now()
        setup_value = 0
        setup_rate = 0
        print(f"ASSIGNMENT: setting up timeline at {setup_time} with initial value {setup_value}, initial rate {setup_rate}")

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        assignment_value = 3

        setup_dt = 50
        assignment_time = setup_time+dt.timedelta(seconds=setup_dt)
        
        new_assignment_impact = Impact(time=assignment_time, type=ImpactType.ASSIGNMENT, value = assignment_value, owner=None)
        print(f"Adding assignment impact at time {assignment_time}")
        timeline.add_impact(new_assignment_impact)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=+1), print_debug=True)
        self.assertEqual(_value, assignment_value)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, assignment_value)

    def test_with_addition(self):
        setup_time = dt.datetime.now()
        setup_value = 1
        setup_rate = 0

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        addition_value = 3

        setup_dt = 50
        addition_time = setup_time+dt.timedelta(seconds=setup_dt)
        
        new_addition_impact = Impact(time=addition_time, type=ImpactType.ADDITION, value = addition_value, owner=None)
        timeline.add_impact(new_addition_impact)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=+1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

    def test_with_rate_addition(self):
        setup_time = dt.datetime.now()
        setup_value = 1
        setup_rate = 0

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        rate_addition_value = 3

        setup_dt = 50
        rate_addition_time = setup_time+dt.timedelta(seconds=setup_dt)
        
        new_rate_addition_impact = Impact(time=rate_addition_time, type=ImpactType.RATE_ADDITION, value = rate_addition_value, owner=None)
        timeline.add_impact(new_rate_addition_impact)

        query_ddt = 30

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=query_ddt), print_debug=True)
        self.assertEqual(_value, setup_value+rate_addition_value*query_ddt)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value)

    def test_with_additions(self):
        setup_time = dt.datetime.now()
        setup_value = 1
        setup_rate = 0

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        addition_value = 3
        rate_addition_value = 3

        setup_dta = 50

        setup_dtr = 70
        
        addition_time = setup_time+dt.timedelta(seconds=setup_dta)

        rate_addition_time = setup_time+dt.timedelta(seconds=setup_dtr)
        
        new_addition_impact = Impact(time=addition_time, type=ImpactType.ADDITION, value = addition_value, owner=None)
        timeline.add_impact(new_addition_impact)

        new_rate_addition_impact = Impact(time=rate_addition_time, type=ImpactType.RATE_ADDITION, value = rate_addition_value, owner=None)
        timeline.add_impact(new_rate_addition_impact)

        query_ddt = 30

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=+1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=query_ddt), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value+rate_addition_value*query_ddt)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

    def test_with_negative_additions(self):
        setup_time = dt.datetime.now()
        setup_value = 1
        setup_rate = 0

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        addition_value = -3
        rate_addition_value = -7

        setup_dta = 20

        setup_dtr = 40
        
        addition_time = setup_time+dt.timedelta(seconds=setup_dta)

        rate_addition_time = setup_time+dt.timedelta(seconds=setup_dtr)
        
        new_addition_impact = Impact(time=addition_time, type=ImpactType.ADDITION, value = addition_value, owner=None)
        timeline.add_impact(new_addition_impact)

        new_rate_addition_impact = Impact(time=rate_addition_time, type=ImpactType.RATE_ADDITION, value = rate_addition_value, owner=None)
        timeline.add_impact(new_rate_addition_impact)

        query_ddt = 15

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=+1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=query_ddt), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value+rate_addition_value*query_ddt)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

    def test_with_negative_additions_and_assignment(self):
        setup_time = dt.datetime.now()
        setup_value = 1
        setup_rate = 0

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        addition_value = -3
        rate_addition_value = -7

        assignment_value=42

        setup_dta = 20

        setup_dtr = 40

        setup_dts = 70
        
        addition_time = setup_time+dt.timedelta(seconds=setup_dta)

        rate_addition_time = setup_time+dt.timedelta(seconds=setup_dtr)
        
        assignment_time = setup_time+dt.timedelta(seconds=setup_dts)

        new_addition_impact = Impact(time=addition_time, type=ImpactType.ADDITION, value = addition_value, owner=None)
        timeline.add_impact(new_addition_impact)

        new_rate_addition_impact = Impact(time=rate_addition_time, type=ImpactType.RATE_ADDITION, value = rate_addition_value, owner=None)
        timeline.add_impact(new_rate_addition_impact)

        new_assignment_impact = Impact(time=assignment_time, type=ImpactType.ASSIGNMENT, value = assignment_value, owner=None)
        timeline.add_impact(new_assignment_impact)

        query_ddt = 15
        assert setup_dtr+query_ddt<setup_dts, "This is not the test you want to run, check the next one"

        query_dts = 15

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=+1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=query_ddt), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value+rate_addition_value*query_ddt)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value+rate_addition_value*((assignment_time-rate_addition_time).total_seconds()-1))

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=query_dts), print_debug=True)
        self.assertEqual(_value, assignment_value+rate_addition_value*query_dts)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, assignment_value)

    def test_with_negative_additions_and_assignment(self):
        setup_time = dt.datetime.now()
        setup_value = 1
        setup_rate = 0

        timeline = Timeline("test_timeline", initial_time=setup_time, initial_value=setup_value, initial_rate=setup_rate)
        
        addition_value = -3
        rate_addition_value = -7

        assignment_value=42

        setup_dta = 20

        setup_dtr = 70

        setup_dts = 40 # Assignment
        
        addition_time = setup_time+dt.timedelta(seconds=setup_dta)

        rate_addition_time = setup_time+dt.timedelta(seconds=setup_dtr)
        
        assignment_time = setup_time+dt.timedelta(seconds=setup_dts)

        new_addition_impact = Impact(time=addition_time, type=ImpactType.ADDITION, value = addition_value, owner=None)
        timeline.add_impact(new_addition_impact)

        new_rate_addition_impact = Impact(time=rate_addition_time, type=ImpactType.RATE_ADDITION, value = rate_addition_value, owner=None)
        timeline.add_impact(new_rate_addition_impact)

        new_assignment_impact = Impact(time=assignment_time, type=ImpactType.ASSIGNMENT, value = assignment_value, owner=None)
        timeline.add_impact(new_assignment_impact)

        query_ddt = 15
        assert setup_dts+query_ddt<setup_dtr, "This is not the test you want to run, check the previous one"

        # Add
        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=+1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        # Rate
        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, assignment_value)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=query_ddt), print_debug=True)
        self.assertEqual(_value, assignment_value+rate_addition_value*query_ddt)

        _value = timeline.get_value_at(rate_addition_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, assignment_value)

        # Assign
        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=-1), print_debug=True)
        self.assertEqual(_value, setup_value+addition_value)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=query_ddt), print_debug=True)
        self.assertEqual(_value, assignment_value)

        _value = timeline.get_value_at(assignment_time+dt.timedelta(seconds=0), print_debug=True)
        self.assertEqual(_value, assignment_value)

if __name__ == '__main__':
    unittest.main()