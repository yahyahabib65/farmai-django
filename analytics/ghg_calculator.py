"""
Greenhouse Gas (GHG) Fluxes Calculator

Implements calculation of agricultural GHG emissions and sequestration
using IPCC Tier 1 and Tier 2 emission factors.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CropParameters:
    """Crop-specific parameters for GHG calculations."""
    name: str
    residue_to_crop_ratio: float  # Ratio of above-ground residue to crop yield
    dry_matter_fraction: float  # Fraction of dry matter in residue
    nitrogen_content: float  # N content of residue (fraction)
    carbon_content: float  # C content (fraction of dry matter)
    below_ground_ratio: float  # Below-ground to above-ground ratio
    burning_fraction: float = 0.0  # Fraction of residue typically burned


# IPCC default values for common crops
CROP_PARAMETERS = {
    'wheat': CropParameters('Wheat', 1.3, 0.89, 0.006, 0.45, 0.24),
    'rice': CropParameters('Rice', 1.4, 0.89, 0.007, 0.42, 0.16),
    'maize': CropParameters('Maize', 1.0, 0.87, 0.006, 0.47, 0.22),
    'cotton': CropParameters('Cotton', 1.5, 0.90, 0.012, 0.45, 0.20),
    'sugarcane': CropParameters('Sugarcane', 0.3, 0.28, 0.004, 0.42, 0.20, 0.1),
    'soybean': CropParameters('Soybean', 2.1, 0.91, 0.008, 0.45, 0.19),
    'potato': CropParameters('Potato', 0.4, 0.22, 0.016, 0.42, 1.00),
    'vegetables': CropParameters('Vegetables', 0.3, 0.10, 0.015, 0.42, 0.50),
    'orchard': CropParameters('Orchard', 0.1, 0.40, 0.010, 0.48, 0.40),
    'default': CropParameters('Default', 1.0, 0.85, 0.007, 0.45, 0.25)
}


# IPCC emission factors
class EmissionFactors:
    """IPCC Tier 1 emission factors."""
    
    # Direct N2O from managed soils (kg N2O-N/kg N applied)
    EF1_SYNTHETIC_N = 0.01  # Synthetic fertilizer
    EF1_ORGANIC_N = 0.01  # Organic fertilizer
    EF1_RESIDUE_N = 0.01  # Crop residue N
    
    # Indirect N2O emissions
    EF4_VOLATILIZATION = 0.01  # From atmospheric deposition
    EF5_LEACHING = 0.0075  # From leaching and runoff
    
    # Fractions
    FRAC_GASF = 0.10  # Synthetic fertilizer N volatilized
    FRAC_GASM = 0.20  # Organic fertilizer N volatilized
    FRAC_LEACH = 0.30  # N lost through leaching
    
    # CH4 from rice paddies (kg CH4/ha/day)
    EF_CH4_RICE = {
        'continuously_flooded': 1.30,
        'single_aeration': 0.60,
        'multiple_aeration': 0.52,
        'rainfed': 0.39,
        'upland': 0.00
    }
    
    # CO2 from fuel use (kg CO2/liter)
    EF_CO2_DIESEL = 2.68
    EF_CO2_PETROL = 2.31
    
    # CO2 from electricity (kg CO2/kWh) - varies by country
    EF_CO2_ELECTRICITY = {
        'pakistan': 0.42,
        'india': 0.82,
        'global_average': 0.50
    }
    
    # N2O to CO2 equivalent (GWP100)
    GWP_N2O = 298
    
    # CH4 to CO2 equivalent (GWP100)
    GWP_CH4 = 25


class GHGCalculator:
    """
    Calculate greenhouse gas emissions from agricultural activities.
    
    Implements IPCC 2006 Guidelines for National Greenhouse Gas Inventories.
    """
    
    def __init__(self, country: str = 'pakistan'):
        self.country = country
        self.ef = EmissionFactors()
    
    def calculate_n2o_from_fertilizer(self,
                                       synthetic_n_kg: float,
                                       organic_n_kg: float = 0,
                                       area_ha: float = 1) -> Dict:
        """
        Calculate N2O emissions from nitrogen fertilizer application.
        
        IPCC Tier 1 approach.
        
        Args:
            synthetic_n_kg: Total synthetic N applied (kg)
            organic_n_kg: Total organic N applied (kg)
            area_ha: Field area in hectares
            
        Returns:
            N2O emissions in kg CO2-eq
        """
        # Direct N2O emissions
        direct_n2o_n = (
            synthetic_n_kg * self.ef.EF1_SYNTHETIC_N +
            organic_n_kg * self.ef.EF1_ORGANIC_N
        )
        
        # Indirect emissions from volatilization
        volatilized_n = (
            synthetic_n_kg * self.ef.FRAC_GASF +
            organic_n_kg * self.ef.FRAC_GASM
        )
        indirect_vol_n2o_n = volatilized_n * self.ef.EF4_VOLATILIZATION
        
        # Indirect emissions from leaching
        total_n = synthetic_n_kg + organic_n_kg
        leached_n = total_n * self.ef.FRAC_LEACH
        indirect_leach_n2o_n = leached_n * self.ef.EF5_LEACHING
        
        # Total N2O-N
        total_n2o_n = direct_n2o_n + indirect_vol_n2o_n + indirect_leach_n2o_n
        
        # Convert N2O-N to N2O
        total_n2o = total_n2o_n * 44/28
        
        # Convert to CO2 equivalent
        total_co2eq = total_n2o * self.ef.GWP_N2O
        
        return {
            'direct_n2o_kg': float(direct_n2o_n * 44/28),
            'indirect_volatilization_n2o_kg': float(indirect_vol_n2o_n * 44/28),
            'indirect_leaching_n2o_kg': float(indirect_leach_n2o_n * 44/28),
            'total_n2o_kg': float(total_n2o),
            'co2_equivalent_kg': float(total_co2eq),
            'per_hectare_co2eq_kg': float(total_co2eq / area_ha) if area_ha > 0 else 0
        }
    
    def calculate_n2o_from_residues(self,
                                    crop_yield_kg: float,
                                    crop_type: str = 'default',
                                    residue_management: str = 'retained',
                                    area_ha: float = 1) -> Dict:
        """
        Calculate N2O emissions from crop residue decomposition.
        
        Args:
            crop_yield_kg: Crop yield in kg
            crop_type: Type of crop
            residue_management: 'retained', 'removed', 'burned'
            area_ha: Field area in hectares
            
        Returns:
            N2O emissions
        """
        params = CROP_PARAMETERS.get(crop_type.lower(), CROP_PARAMETERS['default'])
        
        # Calculate residue production
        above_ground_residue = crop_yield_kg * params.residue_to_crop_ratio
        below_ground_residue = above_ground_residue * params.below_ground_ratio
        
        # Dry matter
        ag_residue_dm = above_ground_residue * params.dry_matter_fraction
        bg_residue_dm = below_ground_residue * params.dry_matter_fraction
        
        # Nitrogen in residues
        if residue_management == 'retained':
            ag_residue_n = ag_residue_dm * params.nitrogen_content
        elif residue_management == 'removed':
            ag_residue_n = 0
        elif residue_management == 'burned':
            ag_residue_n = ag_residue_dm * params.nitrogen_content * 0.1  # 90% lost
        else:
            ag_residue_n = ag_residue_dm * params.nitrogen_content
        
        bg_residue_n = bg_residue_dm * params.nitrogen_content
        
        total_residue_n = ag_residue_n + bg_residue_n
        
        # N2O emissions
        n2o_n = total_residue_n * self.ef.EF1_RESIDUE_N
        n2o = n2o_n * 44/28
        co2eq = n2o * self.ef.GWP_N2O
        
        return {
            'above_ground_residue_kg': float(above_ground_residue),
            'below_ground_residue_kg': float(below_ground_residue),
            'residue_nitrogen_kg': float(total_residue_n),
            'n2o_kg': float(n2o),
            'co2_equivalent_kg': float(co2eq),
            'per_hectare_co2eq_kg': float(co2eq / area_ha) if area_ha > 0 else 0
        }
    
    def calculate_ch4_from_rice(self,
                                 area_ha: float,
                                 cultivation_days: int = 120,
                                 water_regime: str = 'continuously_flooded',
                                 organic_amendment_rate: float = 0) -> Dict:
        """
        Calculate CH4 emissions from flooded rice cultivation.
        
        Args:
            area_ha: Rice paddy area in hectares
            cultivation_days: Growing season length
            water_regime: Water management type
            organic_amendment_rate: Organic amendment (tonnes/ha)
            
        Returns:
            CH4 emissions
        """
        # Base emission factor
        ef_base = self.ef.EF_CH4_RICE.get(water_regime, 1.30)
        
        # Scaling factors
        sf_water = 1.0  # Already in ef_base selection
        
        # Organic amendment scaling (simplified)
        if organic_amendment_rate > 0:
            sf_organic = 1 + organic_amendment_rate * 0.14  # IPCC factor
        else:
            sf_organic = 1.0
        
        # Calculate emissions
        daily_emission = ef_base * sf_water * sf_organic
        total_ch4 = daily_emission * cultivation_days * area_ha
        
        co2eq = total_ch4 * self.ef.GWP_CH4
        
        return {
            'ch4_kg': float(total_ch4),
            'co2_equivalent_kg': float(co2eq),
            'daily_emission_rate_kg_ha': float(daily_emission),
            'cultivation_days': cultivation_days,
            'water_regime': water_regime
        }
    
    def calculate_co2_from_fuel(self,
                                 diesel_liters: float = 0,
                                 petrol_liters: float = 0,
                                 electricity_kwh: float = 0) -> Dict:
        """
        Calculate CO2 emissions from fuel and electricity use.
        
        Args:
            diesel_liters: Diesel consumption
            petrol_liters: Petrol consumption
            electricity_kwh: Electricity consumption
            
        Returns:
            CO2 emissions
        """
        co2_diesel = diesel_liters * self.ef.EF_CO2_DIESEL
        co2_petrol = petrol_liters * self.ef.EF_CO2_PETROL
        
        ef_electricity = self.ef.EF_CO2_ELECTRICITY.get(
            self.country, 
            self.ef.EF_CO2_ELECTRICITY['global_average']
        )
        co2_electricity = electricity_kwh * ef_electricity
        
        total_co2 = co2_diesel + co2_petrol + co2_electricity
        
        return {
            'co2_from_diesel_kg': float(co2_diesel),
            'co2_from_petrol_kg': float(co2_petrol),
            'co2_from_electricity_kg': float(co2_electricity),
            'total_co2_kg': float(total_co2)
        }
    
    def calculate_carbon_sequestration(self,
                                        area_ha: float,
                                        ndvi_mean: float,
                                        crop_type: str = 'default',
                                        practice: str = 'conventional') -> Dict:
        """
        Estimate carbon sequestration potential.
        
        Uses NDVI as proxy for biomass and carbon uptake.
        
        Args:
            area_ha: Field area
            ndvi_mean: Mean NDVI value
            crop_type: Crop type
            practice: 'conventional', 'no_till', 'cover_crop', 'agroforestry'
            
        Returns:
            Carbon sequestration estimate
        """
        params = CROP_PARAMETERS.get(crop_type.lower(), CROP_PARAMETERS['default'])
        
        # Estimate biomass from NDVI (simplified relationship)
        # This is a rough approximation - actual values vary significantly
        biomass_factor = 10000  # kg DM/ha at NDVI=1
        estimated_biomass = ndvi_mean * biomass_factor * area_ha
        
        # Carbon in biomass
        carbon_in_biomass = estimated_biomass * params.carbon_content
        
        # Sequestration rate depends on practice
        sequestration_rates = {
            'conventional': 0.1,  # 10% of C may be sequestered
            'no_till': 0.25,  # Better soil C retention
            'cover_crop': 0.35,  # Additional C input
            'agroforestry': 0.50  # Significant sequestration
        }
        
        seq_rate = sequestration_rates.get(practice, 0.1)
        sequestered_c = carbon_in_biomass * seq_rate
        
        # Convert to CO2 equivalent
        co2_sequestered = sequestered_c * 44/12
        
        return {
            'estimated_biomass_kg': float(estimated_biomass),
            'carbon_in_biomass_kg': float(carbon_in_biomass),
            'sequestration_rate': float(seq_rate),
            'sequestered_carbon_kg': float(sequestered_c),
            'co2_equivalent_sequestered_kg': float(co2_sequestered),
            'per_hectare_co2_sequestered_kg': float(co2_sequestered / area_ha) if area_ha > 0 else 0,
            'practice': practice
        }
    
    def calculate_total_ghg_balance(self,
                                    area_ha: float,
                                    crop_type: str,
                                    crop_yield_kg: float,
                                    synthetic_n_kg: float,
                                    organic_n_kg: float = 0,
                                    diesel_liters: float = 0,
                                    electricity_kwh: float = 0,
                                    is_rice: bool = False,
                                    rice_water_regime: str = 'continuously_flooded',
                                    rice_cultivation_days: int = 120,
                                    ndvi_mean: float = 0.5,
                                    practice: str = 'conventional') -> Dict:
        """
        Calculate complete GHG balance for a field.
        
        Args:
            All parameters from individual calculations
            
        Returns:
            Complete GHG balance with net emissions
        """
        # Calculate all emission sources
        fertilizer = self.calculate_n2o_from_fertilizer(
            synthetic_n_kg, organic_n_kg, area_ha
        )
        
        residues = self.calculate_n2o_from_residues(
            crop_yield_kg, crop_type, 'retained', area_ha
        )
        
        fuel = self.calculate_co2_from_fuel(
            diesel_liters, 0, electricity_kwh
        )
        
        sequestration = self.calculate_carbon_sequestration(
            area_ha, ndvi_mean, crop_type, practice
        )
        
        # Rice-specific CH4
        if is_rice or crop_type.lower() == 'rice':
            rice = self.calculate_ch4_from_rice(
                area_ha, rice_cultivation_days, rice_water_regime
            )
            rice_emissions = rice['co2_equivalent_kg']
        else:
            rice = None
            rice_emissions = 0
        
        # Total emissions
        total_emissions = (
            fertilizer['co2_equivalent_kg'] +
            residues['co2_equivalent_kg'] +
            fuel['total_co2_kg'] +
            rice_emissions
        )
        
        # Net balance
        net_emissions = total_emissions - sequestration['co2_equivalent_sequestered_kg']
        
        return {
            'total_emissions_co2eq_kg': float(total_emissions),
            'total_sequestration_co2eq_kg': float(sequestration['co2_equivalent_sequestered_kg']),
            'net_emissions_co2eq_kg': float(net_emissions),
            'per_hectare_net_emissions_kg': float(net_emissions / area_ha) if area_ha > 0 else 0,
            'per_kg_yield_emissions': float(net_emissions / crop_yield_kg) if crop_yield_kg > 0 else 0,
            'is_net_sink': net_emissions < 0,
            'breakdown': {
                'fertilizer': fertilizer,
                'residues': residues,
                'fuel': fuel,
                'rice': rice,
                'sequestration': sequestration
            },
            'field_info': {
                'area_ha': area_ha,
                'crop_type': crop_type,
                'practice': practice
            }
        }


def calculate_field_ghg(field_id: int,
                        synthetic_n_kg: float = None,
                        diesel_liters: float = None) -> Dict:
    """
    Calculate GHG balance for a specific field.
    """
    from core.models import FieldBoundary
    from analytics.models import AnalyticsResult
    
    try:
        field = FieldBoundary.objects.get(id=field_id)
    except FieldBoundary.DoesNotExist:
        return {'error': f'Field {field_id} not found'}
    
    # Get field parameters
    area_ha = field.area_hectares or 1.0
    crop_type = field.crop_type or 'default'
    
    # Get latest NDVI
    latest_result = AnalyticsResult.objects.filter(
        field=field,
        avg_ndvi__isnull=False
    ).order_by('-analysis_date').first()
    
    ndvi_mean = latest_result.avg_ndvi if latest_result else 0.5
    
    # Estimate yield from NDVI (rough approximation)
    # Actual yield would come from manual records
    estimated_yield = area_ha * 3000 * ndvi_mean  # ~3 tonnes/ha at NDVI=1
    
    # Default inputs if not provided
    if synthetic_n_kg is None:
        # Typical N application: 150 kg/ha
        synthetic_n_kg = area_ha * 150
    
    if diesel_liters is None:
        # Typical diesel use: 100 L/ha
        diesel_liters = area_ha * 100
    
    calculator = GHGCalculator(country='pakistan')
    
    result = calculator.calculate_total_ghg_balance(
        area_ha=area_ha,
        crop_type=crop_type,
        crop_yield_kg=estimated_yield,
        synthetic_n_kg=synthetic_n_kg,
        diesel_liters=diesel_liters,
        ndvi_mean=ndvi_mean,
        is_rice=(crop_type.lower() == 'rice')
    )
    
    result['field_id'] = field_id
    result['field_name'] = field.name
    
    return result
