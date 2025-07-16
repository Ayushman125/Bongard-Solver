# Folder: bongard_solver/src/utils/
# File: compute_temperature.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_temperature(
    current_step: int,
    max_steps: int,
    initial_temperature: float = 1.0,
    final_temperature: float = 0.1,
    annealing_type: str = 'linear'
) -> float:
    """
    Computes the current system temperature based on the progress of the simulation.
    Temperature can influence the randomness and exploration vs. exploitation trade-off
    in emergent systems (e.g., in the urgency of codelets).

    Args:
        current_step (int): The current step or iteration in the simulation.
        max_steps (int): The total number of steps/iterations for annealing.
        initial_temperature (float): The starting temperature.
        final_temperature (float): The ending temperature.
        annealing_type (str): The type of annealing schedule ('linear', 'cosine', 'exponential').

    Returns:
        float: The computed temperature for the current step.
    """
    if max_steps <= 0:
        logger.warning("Max steps is non-positive. Returning initial temperature.")
        return initial_temperature
    
    if current_step >= max_steps:
        return final_temperature

    progress = current_step / max_steps
    temperature = initial_temperature

    if annealing_type == 'linear':
        temperature = initial_temperature - progress * (initial_temperature - final_temperature)
    elif annealing_type == 'cosine':
        # Cosine annealing from initial_temperature to final_temperature
        temperature = final_temperature + 0.5 * (initial_temperature - final_temperature) * \
                      (1 + (progress * 3.1415926535).cos()) # pi
    elif annealing_type == 'exponential':
        # Exponential decay
        decay_rate = (final_temperature / initial_temperature)**(1 / max_steps)
        temperature = initial_temperature * (decay_rate ** current_step)
    else:
        logger.warning(f"Unknown annealing type '{annealing_type}'. Using linear annealing.")
        temperature = initial_temperature - progress * (initial_temperature - final_temperature)

    # Ensure temperature stays within bounds
    temperature = max(final_temperature, min(initial_temperature, temperature))
    
    logger.debug(f"Current Temperature (step {current_step}/{max_steps}): {temperature:.4f}")
    return temperature

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running compute_temperature.py example.")

    max_steps_example = 100
    initial_temp = 1.0
    final_temp = 0.01

    print("\n--- Linear Annealing ---")
    for step in [0, 25, 50, 75, 100, 120]:
        temp = compute_temperature(step, max_steps_example, initial_temp, final_temp, 'linear')
        print(f"Step {step}: Temperature = {temp:.4f}")

    print("\n--- Cosine Annealing ---")
    for step in [0, 25, 50, 75, 100, 120]:
        temp = compute_temperature(step, max_steps_example, initial_temp, final_temp, 'cosine')
        print(f"Step {step}: Temperature = {temp:.4f}")

    print("\n--- Exponential Annealing ---")
    for step in [0, 25, 50, 75, 100, 120]:
        temp = compute_temperature(step, max_steps_example, initial_temp, final_temp, 'exponential')
        print(f"Step {step}: Temperature = {temp:.4f}")

    print("\n--- Edge Cases ---")
    print(f"Max steps 0: {compute_temperature(0, 0)}")
    print(f"Current step > max steps: {compute_temperature(200, 100)}")
