import type { ResumeView, TargetRoleView } from "../api/client";

export interface ResumeRoleGroup {
  roleId: string;
  title: string;
  resumes: ResumeView[];
}

/**
 * The library by target role, in the roles' own priority order.
 *
 * Every role is a group even with no resume yet, so it can be imported into.
 * A resume whose role is missing from the list (it should not happen: a
 * resume always has a role) still shows, in a group of its own, rather than
 * silently disappearing from the page.
 */
export function groupResumesByRole(resumes: ResumeView[], roles: TargetRoleView[]): ResumeRoleGroup[] {
  const groups: ResumeRoleGroup[] = roles.map((role) => ({
    roleId: role.id,
    title: role.title,
    resumes: resumes.filter((resume) => resume.target_role_id === role.id),
  }));
  const known = new Set(roles.map((role) => role.id));
  for (const resume of resumes) {
    if (known.has(resume.target_role_id)) continue;
    known.add(resume.target_role_id);
    groups.push({
      roleId: resume.target_role_id,
      title: resume.target_role,
      resumes: resumes.filter((item) => item.target_role_id === resume.target_role_id),
    });
  }
  return groups;
}
