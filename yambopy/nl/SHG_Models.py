

import numpy as np
from yambo.units import FREE_SPACE_PERM, speed_of_light_SI, hbar, electron_charge_SI

class Stack:
    
    def __init__(self, material_2D, film, substrate, film_thickness, h_2D):
        self.material_2D = material_2D
        self.film = film
        self.substrate = substrate
        self.d = film_thickness
        self.h_2D = h_2D
        self.n0 = 1  # air
        
    def usable_omega(self, omega_eV):
        "Energies where every layer has data at both w and 2w"
        w = np.asarray(omega_eV, float)
        usable = np.ones(len(W), bool)
        for mat in (self.material_2D, self.film, self.substrate):
            lo, hi = mat.wl_range_eV()
            usable &= (w >= lo) & (w <= hi) & (2*w >= lo) & (2*w <= hi)
        return usable 
    
    def _r_ij(ni, nj):
        "Fresnel reflection coefficient between two media."
        return (ni - nj)/(ni + nj)

    def _Rs(self, wavelength, n1, n2):
        "Film+substrate reflection without the 2D layer, including film"
        interference.  Song et al. Eq. (4)."
        w_tilde = 2*np.pi/wavelength
        r01 = self._r_ij(self.n0, n1)
        r12 = self._r_ij(n1, n2)
        phase = np.exp(2j*w_tilde*n1*self.d)
        return (r01 + r12*phase)/(1 + r01*r12*phase)

    def _R_total(self, wavelength, n2D, n1, n2):
       "Total reflection of air/material_2D/film/substrate.  Song et al. Eq. (3)."
        w_tilde = 2*np.pi/wavelength
        eta = -1j*self.h_2D*w_tilde*(n2D**2 - 1)/2
        r = -eta/(1 + eta)
        t = 1/(1 + eta)
        Rs = self._Rs(wavelength, n1, n2)
        return r + (Rs*t**2)/(1 - Rs*r)
    
    def structure_factor(self, omega_eV):
        "beta = (1 + R_w)^2 (1 + R_2w), Song et al. Eq. (2)."
        w = np.asarray(omega_eV, float)
        usable = self.usable_omega(w)
        beta = np.full(len(w), np.nan, dtype=complex)
        if not usable.any():
            print("Stack: no omega usable by all materials at omega and 2*omega")
            return beta
        w_valid = w[usable]
        wavelength = HC_EV_M/w_valid
        R_w = self._R_total(wavelength, self.material_2D.complex_index(w_valid), self.film.complex_index(w_valid), self.substrate.complex_index(w_valid))
        R_2w = self._R_total(wavelength/2,self.material_2D.complex_index(2*w_valid), self.film.complex_index(2*w_valid), self.substrate.complex_index(2*w_valid))
        beta[usable] = (1 + R_w)**2 * (1 + R_2w)
        return beta
    
    def shg_intensity(self, omega_eV, chi2_sheet, I_incident):
        "SHG intensity, Song et al. Eq. (1) for chi2_sheet in m^2/V. In the absence of dielectric film, reduces to Woodward model, while allowing absorbing substrate.
        w = np.asarray(omega_eV, float)
        wavelength = HC_EV_M/w
        beta = self.structure_factor(w)
        chi_term = np.abs(2*np.pi*np.asarray(chi2_sheet)/wavelength)**2
        return (1/(2*FREE_SPACE_PERM*speed_of_light_SI)) * np.abs(beta)**2 \ * chi_term * I_incident**2

    def __str__(self):
        return ("\n * * * Stack * * *\n"
                "2D material : %s\nFilm        : %s (d = %.1f nm)\n"
                "Substrate   : %s\nh_2D        : %.2f nm\n"
                % (getattr(self.material_2D, 'name', '?'),
                   getattr(self.film, 'name', '?'), self.d*1e9,
                   getattr(self.substrate, 'name', '?'), self.h_2D*1e9))
    
    
    
  class ClarkModel:
      "Come back to comment here on the bulk model and where it comes from"
      def __init__(self, material_2D, substrate, h_2D):
          self.material_2D = material_2D
          self.substrate = substrate
          self.h_2D = h_2D
          
      def _omega_rad(self, omega_eV):
          return np.asarray(omega_eV, float) * electron_charge_SI / hbar
      
        
      def usable_omega(self, omega_eV):
          w = np.asarray(omega_eV, float)
          lo, hi = self.material_2D.wl_range_eV()
          return (w >= lo) & (w <= hi) & (2*w >= lo) & (2*w <= hi)
      
      def bulk_intensity(self, omega_eV, chi2_bulk, I_incident):
          w = np.asarray(omega_eV, float)
          usable = self.usable_omega(w)
          I = np.full(len(w), np.nan, float)
          if not usable.any():
              print("ClarkModel: no omega usable by the 2D material at omega and 2*omega");
              return
          w_valid = w[usable]
          omega = self._omega_rad(w_valid)
          n_w = np.real(np.asarray(self.material_2D.complex_index(w_valid)))
          n_2w = np.real(np.asarray(self.material_2D.complex_index(2*w_valid)))
          chi = np.asarray(chi2_bulk)[usable] if np.ndim(chi2_bulk) else chi2_bulk
          num = omega**2 * np.abs(chi)**2 * self.h_2D**2 * I_incident**2
          den = 2 * n_w**2 * n_2w * FREE_SPACE_PERM * speed_of_light_SI**3
          I[usable] = num/den
          return I
      
      def __str__(self):
          return  (f"\n * * * ClarkModel * * *\n"
                f"2D material : {getattr(self.material_2D,'name','?')}\n"
                f"Substrate   : {getattr(self.substrate,'name','?')}\n"
                f"h_2D        : {self.h_2D*1e9:.2f} nm\n")
          

  class WoodwardModel:
      "Comment on this sheet model"
      def __init__(self, material_2D, substrate, h_2D):
          self.material_2D = material_2D
          self.substrate = substrate
          self.h_2D = h_2D
          
      def _omega_rad(self, omega_eV):
          return np.asarray(omega_eV, float) * electron_charge_SI / hbar
      
      def usable_omega(self, omega_eV):
          w = np.asarray(omega_eV, float)
          lo, hi = self.substrate.wl_range_eV()
          return (w >= lo) & (w <= hi)
      
      def sheet_intensity(self, omega_eV, chi2_sheet, I_incident):
          w = np.asarray(omega_eV, float)
          usable = self.usable_omega(w)
          I = np.full(len(w), np.nan, float)
          if not usable.any():
              print("WoodwardModel: no omega values usable by the substrate");
              return I
          w_valid = w[usable]
          omega = self._omega_rad(w_valid)
          n_complex = np.asarray(self.substrate.complex_index(w_valid))
          if np.any(np.abs(np.imag(n_complex)) > 0.01):
              print("Woodward sheet model assumes a transparent substrate, but this one absorbs (k>0). Use the structure factor model instead")
          n = np.real(n_complex)
          chi = np.asarray(chi2_sheet)[usable] if np.ndim(chi2_sheet) else chi2_sheet
          num = 32 * omega**2 * np.abs(chi)**2 * I_incident**2 
          den = (n + 1)**6 * FREE_SPACE_PERM * speed_of_light_SI**3
          I[usable] = num / den
          return I
      
        def __str__(self):
            return (f"\n * * * WoodwardModel * * *\n"
                f"2D material : {getattr(self.material_2D,'name','?')}\n"
                f"Substrate   : {getattr(self.substrate,'name','?')}\n"
                f"h_2D        : {self.h_2D*1e9:.2f} nm\n")
            
    
       
   