"""
Composable model implementation that combines components.
"""

import pymc as pm

from ..changepoint_model import ChangepointModel


class ComposableModel(ChangepointModel):
    """Composable changepoint model that combines different components.

    This model allows flexible combination of:
    - Changepoint types (fixed vs Dirichlet process)
    - Change types (sigmoid vs step)
    - Emission types (Poisson vs Gaussian)
    - Likelihood types (Poisson vs Gaussian)
    """

    def __init__(self, data_array, changepoint_component, emission_component,
                 likelihood_component, **kwargs):
        """
        Args:
            data_array: Input data array
            changepoint_component: ChangepointComponent instance
            emission_component: EmissionComponent instance
            likelihood_component: LikelihoodComponent instance
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.data_array = data_array
        self.changepoint_component = changepoint_component
        self.emission_component = emission_component
        self.likelihood_component = likelihood_component

    def generate_model(self):
        """Generate the composable PyMC model.

        Returns:
            pymc model: Model combining all components
        """
        data_array = self.data_array
        time_length = data_array.shape[-1]

        with pm.Model() as model:
            # Generate changepoints and weights
            tau, weight_stack = self.changepoint_component.generate_changepoints(
                time_length)

            # Generate emissions
            emissions = self.emission_component.generate_emissions(data_array)

            # Generate likelihood
            observation = self.likelihood_component.generate_likelihood(
                data_array, emissions, weight_stack)

        return model

    def test(self):
        """Test the composable model with synthetic data."""
        from ..changepoint_model import gen_test_array

        # Determine data type based on likelihood component
        if hasattr(self.likelihood_component, '__class__'):
            if 'Poisson' in self.likelihood_component.__class__.__name__:
                data_type = 'poisson'
            else:
                data_type = 'normal'
        else:
            data_type = 'poisson'  # Default

        # Determine number of states
        if hasattr(self.changepoint_component, 'n_states'):
            n_states = self.changepoint_component.n_states
        elif hasattr(self.changepoint_component, 'max_states'):
            # Use 3 for testing
            n_states = min(3, self.changepoint_component.max_states)
        else:
            n_states = 3

        # Generate test data matching the original data shape
        if len(self.data_array.shape) == 3:
            test_data = gen_test_array(
                (5, 10, 100), n_states=n_states, type=data_type)
        elif len(self.data_array.shape) == 2:
            test_data = gen_test_array(
                (10, 100), n_states=n_states, type=data_type)
        else:
            test_data = gen_test_array(100, n_states=n_states, type=data_type)

        # Create test model with same components but test data
        test_model = ComposableModel(
            test_data,
            self.changepoint_component,
            self.emission_component,
            self.likelihood_component
        )

        model = test_model.generate_model()

        # Run minimal inference to verify model works
        with model:
            # Just do a few iterations to test functionality
            inference = pm.ADVI()
            approx = pm.fit(n=10, method=inference)
            trace = approx.sample(draws=10)

        # Check if expected variables are in the trace
        assert "obs" in trace.varnames

        print(f"Test for ComposableModel passed")
        return True
